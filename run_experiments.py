import os
import torch
import numpy as np
import xarray as xr
from torch import optim
import logging
import csv
from experiments import GROUPS
from DA_framework import L96_DA_Experiment as L96_DA
from DA_framework import QG1_DA_Experiment as QG1_DA
from DA_framework import DA_UNet
from models import L96Model, QG1Model

logging.getLogger().setLevel(logging.WARNING)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ← Change this one line to switch experiment groups
ACTIVE_GROUP = 'all_forces_EnKF'

group         = GROUPS[ACTIVE_GROUP]
experiments   = group['experiments']
widths        = group['widths']
learning_rate = group['learning_rate']
dropout       = group['dropout']

n_epochs = 100


# =============================================================================
# Helper functions
# =============================================================================

def get_DA_experiment(exp, name):
    """Instantiate the correct DA experiment subclass based on model_id."""
    model_id = exp.get('model_id')
    if model_id == 'L96':
        return L96_DA(**exp)
    elif model_id in ['QG', 'QG2']:
        return QG1_DA(**exp)
    else:
        raise ValueError(f"Unknown model_id '{model_id}'. "
                         f"Add a corresponding DA experiment class and register it here.")


def get_base_dir(exp):
    """Derive the base data directory from model_id."""
    return f'./data/{exp.get("model_id")}'


def get_unet_input(x_norm, exp):
    """
    Reshape normalised state data into the correct input tensor shape for DA_UNet.
        L96  : (samples, 1, N)
        QG   : (samples, 1, nx, ny)
        QG2  : (samples, 2, nx, ny)  — two-layer QG, in_channels=2
    """
    model_id = exp.get('model_id')
    if model_id == 'L96':
        return x_norm[:, np.newaxis, :]
    elif model_id == 'QG':
        nx = exp.get('nx', 32)
        ny = exp.get('ny', 32)
        return x_norm.reshape(-1, 1, nx, ny)
    elif model_id == 'QG2':
        nx = exp.get('nx', 32)
        ny = exp.get('ny', 32)
        return x_norm.reshape(-1, 2, nx, ny)
    else:
        raise ValueError(f"Unknown model_id '{model_id}'. "
                         f"Add an input reshaping case for this model.")


def get_spatial_dims(exp):
    """Return the correct spatial_dims for DA_UNet based on model_id."""
    model_id = exp.get('model_id')
    if model_id == 'L96':
        return 1
    elif model_id in ['QG', 'QG2']:
        return 2
    else:
        raise ValueError(f"Unknown model_id '{model_id}'. "
                         f"Add a spatial_dims case for this model.")


def get_unet_in_channels(exp):
    """Return the number of input channels for DA_UNet based on model_id."""
    model_id = exp.get('model_id')
    if model_id == 'L96':
        return 1
    elif model_id == 'QG':
        return 1
    elif model_id == 'QG2':
        return 2
    else:
        raise ValueError(f"Unknown model_id '{model_id}'. "
                         f"Add an in_channels case for this model.")


def build_file_name(exp, method=None):
    """Reconstruct the file_name() string used by da_framework."""
    m = method or exp['DA_method']
    return (f'{m}_N{exp["N_DA"]}_ens{exp["nens"]}_freq{exp["obs_freq"]}'
            f'_nobs{exp["nobs"]}_err{exp["obs_err"]:.1e}')


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':

    # Quick sanity check using the first experiment's model
    first_exp = experiments[0]
    model_id  = first_exp.get('model_id')
    if model_id == 'L96':
        test_model = L96Model(N=first_exp['N_truth'], F=first_exp['F'], dt=first_exp['dt'])
        test_model.x = first_exp['F'] * np.ones(first_exp['N_truth'])
        test_model.x[0] += 0.01
        print(f"x[0:5] before: {test_model.x[:5]}")
        test_model.step_forward()
        print(f"x[0:5] after:  {test_model.x[:5]}")

    for i,exp in enumerate(experiments):
        print(f'\n{"="*50}')
        print(f'Running Experiment: {exp["exp_id"]}  [{i+1}/{len(experiments)}]')
        print(f'{"="*50}')

        N        = exp['N_DA']
        base_dir = get_base_dir(exp)
        DA_train = get_DA_experiment(exp, exp['exp_id'])

        # # ── Sanity check truth  ─────────────────────────────────────────
        # print('\n>>> Checking truth diagnostics...')
        # DA_train.diagnose_truth(steps=exp['steps_truth'])
        # # ────────────────────────────────────────────────────────────────

        # ── Phase 1: Generate training data via Kalman Filter ──────────────
        mean_file_path = f'{base_dir}/{exp["exp_id"]}/EnsMean_{build_file_name(exp)}.nc'

        if not os.path.exists(mean_file_path):
            print('\n>>> Phase 1: Generate Training Data via Kalman Filter')

            truth_file = (f'{base_dir}/Truth_N{exp["N_truth"]}'
                          f'_F{exp["F"]}_dt{exp["dt"]}_{exp["steps_truth"]}steps.nc')
            if not os.path.exists(truth_file):
                print('  Generating truth trajectory...')
                DA_train.generate_truth(steps=exp['steps_truth'], spin_up = exp['steps_spin_up'])
            else:
                print('  Truth file already exists, skipping.')

            DA_train.generate_obs(steps=exp['steps_truth'], save_netcdf=True)
            DA_train.run_exp(DA_steps=exp['steps_train'], truth_steps=exp['steps_truth'],ic_seed=0)
        else:
            print('\n>>> Phase 1: SKIPPING (data already exists)')
        print(f'Phase 1 complete for experiment {exp["exp_id"]}')

        # ── Phases 2 & 3: Preprocess + Train UNet ──────────────────────────
        spatial_dims = get_spatial_dims(exp)
        in_channels  = get_unet_in_channels(exp)
        unet_model   = DA_UNet(N=N, width=widths[i], dropout=dropout[i],
                               spatial_dims=spatial_dims,
                               in_channels=in_channels).to(DEVICE).float()
        train_file   = f'{base_dir}/{exp["exp_id"]}/unet_epoch{n_epochs}.pt'

        if not os.path.exists(train_file):
            print('\n>>> Phase 2: Preprocess Data')

            B_ens_ds = xr.open_dataset(f'{base_dir}/{exp["exp_id"]}/B_ens_{build_file_name(exp)}.nc')
            mean_ds  = xr.open_dataset(mean_file_path).load()

            spin_up     = int(len(B_ens_ds.cycle) * 0.2)
            decorr_skip = max(1, int(1 / exp['dt']))

            B_ens  = B_ens_ds.B_ens.isel(cycle=slice(spin_up, None, decorr_skip))
            x_data = mean_ds.x.isel(time=slice(spin_up, None, decorr_skip))

            B_std = B_ens.std().values
            x_std = x_data.std().values

            xr.Dataset({'B_std': xr.DataArray(B_std),
                        'x_std': xr.DataArray(x_std)}).to_netcdf(
                            f'{base_dir}/{exp["exp_id"]}/std.nc')

            B_data = (B_ens.values / B_std).astype(np.float32)
            x_norm = (x_data.values / x_std).astype(np.float32)

            x_unet = get_unet_input(x_norm, exp)    # model-specific reshape
            B_unet = B_data                          # (samples, N, N) for all models

            print(f'  Samples: {B_data.shape[0]} | B mean: {np.mean(np.abs(B_data)):.4f} | '
                  f'x mean: {np.mean(np.abs(x_norm)):.4f}')

            print(f'\n>>> Phase 3: Training UNet  (epochs={n_epochs})')

            n_total       = x_unet.shape[0]
            n_train_split = int(n_total * 0.8)
            print(f'  Total samples: {n_total}  |  Train: {n_train_split}  |  Val: {n_total - n_train_split}')

            x_train = torch.from_numpy(x_unet[:n_train_split]).float()
            B_train = torch.from_numpy(B_unet[:n_train_split]).float()
            x_valid = torch.from_numpy(x_unet[n_train_split:]).float()
            B_valid = torch.from_numpy(B_unet[n_train_split:]).float()

            loader_params = {'batch_size': 128, 'num_workers': 2, 'shuffle': True}
            train_loader  = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(x_train, B_train), **loader_params)
            valid_loader  = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(x_valid, B_valid), **loader_params)

            criterion = torch.nn.MSELoss()
            optimizer = optim.Adam(unet_model.parameters(), lr=learning_rate[i])

            train_losses, valid_losses = [], []
            best_valid_loss, best_epoch = float('inf'), 0

            for epoch in range(1, n_epochs + 1):
                unet_model.train()
                train_loss = 0.0
                for x_batch, B_batch in train_loader:
                    x_batch, B_batch = x_batch.to(DEVICE), B_batch.to(DEVICE)
                    optimizer.zero_grad()
                    loss = criterion(unet_model(x_batch), B_batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                train_loss /= len(train_loader)
                train_losses.append(train_loss)

                unet_model.eval()
                valid_loss = 0.0
                with torch.no_grad():
                    for x_batch, B_batch in valid_loader:
                        x_batch, B_batch = x_batch.to(DEVICE), B_batch.to(DEVICE)
                        valid_loss += criterion(unet_model(x_batch), B_batch).item()
                valid_loss /= len(valid_loader)
                valid_losses.append(valid_loss)

                if valid_loss < best_valid_loss:
                    best_valid_loss, best_epoch = valid_loss, epoch
                    torch.save(unet_model.state_dict(),
                               f'{base_dir}/{exp["exp_id"]}/unet_best.pt')

                if epoch % 10 == 0:
                    print(f'  Epoch {epoch:3d}/{n_epochs} | train: {train_loss:.6f} | val: {valid_loss:.6f}')
                    torch.save(unet_model.state_dict(),
                               f'{base_dir}/{exp["exp_id"]}/unet_epoch{epoch}.pt')

            print(f'  Best val loss: {best_valid_loss:.6f} at epoch {best_epoch}')

            with open(f'{base_dir}/{exp["exp_id"]}/losses.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['train_loss', 'valid_loss'])
                writer.writerows(zip(train_losses, valid_losses))
            print('  Losses saved.')
        else:
            print('\n>>> Phases 2 & 3: SKIPPING (UNet already trained)')

        # ── Phase 3.5: Save UNet-predicted B matrices ───────────────────────
        b_pred_file = f'{base_dir}/{exp["exp_id"]}/B_unet_predicted.nc'
        if not os.path.exists(b_pred_file):
            print('\n>>> Phase 3.5: Saving UNet Predicted B Matrices')
            ml_std_ds = xr.open_dataset(f'{base_dir}/{exp["exp_id"]}/std.nc')
            unet_model.load_state_dict(
                torch.load(f'{base_dir}/{exp["exp_id"]}/unet_best.pt', map_location=DEVICE))
            unet_model.eval()

            all_B_pred = []
            with torch.no_grad():
                x_tensor = torch.from_numpy(x_unet).float().to(DEVICE)
                for start in range(0, len(x_tensor), 128):
                    B_pred = (unet_model(x_tensor[start:start+128]).cpu().numpy()
                              * ml_std_ds.B_std.values)
                    all_B_pred.append(B_pred)

            all_B_pred = np.concatenate(all_B_pred, axis=0)
            xr.DataArray(all_B_pred, dims=['cycle', 'dim', 'dim_d']).to_netcdf(b_pred_file)
            print(f'  Saved {len(all_B_pred)} predicted B matrices.')
        else:
            print('\n>>> Phase 3.5: SKIPPING (B matrices already saved)')

        # ── Phase 4: Run UNetKF ─────────────────────────────────────────────
        print('\n>>> Phase 4: Running UNetKF')

        model_file = f'{base_dir}/{exp["exp_id"]}/unet_best.pt'
        if not os.path.exists(model_file):
            model_file = f'{base_dir}/{exp["exp_id"]}/unet_epoch{n_epochs}.pt'
        unet_model.load_state_dict(torch.load(model_file, map_location=DEVICE))
        unet_model.eval()
        ml_std_ds = xr.open_dataset(f'{base_dir}/{exp["exp_id"]}/std.nc')

        DA_unet = get_DA_experiment({**exp,
            'nens':      1,
            'DA_method': 'UNetKF',
            'save_B':    False,
            'inflate':   1.0,
        }, exp["exp_id"])

        DA_unet.run_exp(
            DA_steps=exp["steps_train"],
            truth_steps = exp["steps_truth"],
            ic_seed=0,
            output_str=f'/{exp["exp_id"]}_UNetKF',
            ml_model=unet_model,
            ml_std_ds=ml_std_ds,
        )

        print(f'\nExperiment {exp["exp_id"]} complete.')