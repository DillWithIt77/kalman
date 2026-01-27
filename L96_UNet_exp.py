"""
Lorenz 96 UNetKF Training and Experiment Script

This script implements the UNet Kalman Filter approach from Lu (2024) 
"U-Net Kalman Filter (UNetKF): An Example of Machine Learning-assisted 
Ensemble Data Assimilation" for the Lorenz 96 model.

KEY DIFFERENCES FROM THE PAPER (QG MODEL):
1. Model: L96 (1D, 40 variables) vs QG (2D, 32x32 or 128x128 grid)
2. Covariances: Using FULL B matrices (40x40) vs LOCALIZED patches (16x16)
3. Architecture: 1D convolutions vs 2D convolutions
4. Normalization: Using std (as in paper) not range

IMPORTANT CONSIDERATIONS:
- Paper uses LOCALIZED covariance matrices to reduce storage/computation
  We use FULL matrices for L96 since it's smaller (40x40 vs 32x32x2 layers)
- Paper normalizes by standard deviation (we do this)
- Paper uses ~200 epochs and tracks validation loss for early stopping
- Paper finds optimal model at epoch with MINIMUM validation loss
- Paper uses relaxation factors in training EnKF (0.45-0.6)
- Paper uses NO inflation for single-member UNetKF

TO MAKE THIS MORE LIKE THE PAPER:
- Could implement localized B matrices (e.g., 8x8 patches around each point)
- Would need to modify data preprocessing and UNet architecture
- Localization helps with larger models and reduces overfitting
"""

import os
import torch
import numpy as np
import xarray as xr
import L96_DA_core as L96  # Your Lorenz 96 DA module
# import ML_core as ML
from torch import optim
from matplotlib import pyplot as plt
import logging
import torch.nn as nn
import torch.nn.functional as F
import csv
from matplotlib.animation import FuncAnimation

###cut down on what is printed
logging.getLogger().setLevel(logging.WARNING)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = './data/lorenz96'

#-----------------------#
# EXPERIMENT DEFINITIONS
#-----------------------#

# Experiment with 20 ensemble members
DA_exp_L20 = {
    'N_truth': 40,
    'N_DA': 40,
    'nens': 20,
    'DA_method': 'EnKF',
    'obs_freq': 4,          # Observe every 4 time steps
    'obs_err': 1.0,         # Observation error std
    'nobs': 20,             # Number of observations per cycle
    'loc_radius': 5.0,      # Localization radius
    'DA_freq': 4,           # DA cycle frequency
    'save_B': True,         # Save B matrices for training
    'use_localization': False,  # Use FULL B matrix (no localization)
    'inflate': [1.05, 0.0],  # [prior_inflation, relaxation]
    'F': 8.0,               # Lorenz 96 forcing
    'dt': 0.01}             # Time step

DA_exp_L20_w16 = {
    'N_truth': 40,
    'N_DA': 40,
    'nens': 20,
    'DA_method': 'EnKF',
    'obs_freq': 4,          # Observe every 4 time steps
    'obs_err': 1.0,         # Observation error std
    'nobs': 20,             # Number of observations per cycle
    'loc_radius': 5.0,      # Localization radius
    'DA_freq': 4,           # DA cycle frequency
    'save_B': True,         # Save B matrices for training
    'use_localization': False,  # Use FULL B matrix (no localization)
    'inflate': [1.05, 0.0],  # [prior_inflation, relaxation]
    'F': 8.0,               # Lorenz 96 forcing
    'dt': 0.01}             # Time step

DA_exp_L20_w4 = {
    'N_truth': 40,
    'N_DA': 40,
    'nens': 20,
    'DA_method': 'EnKF',
    'obs_freq': 4,          # Observe every 4 time steps
    'obs_err': 1.0,         # Observation error std
    'nobs': 20,             # Number of observations per cycle
    'loc_radius': 5.0,      # Localization radius
    'DA_freq': 4,           # DA cycle frequency
    'save_B': True,         # Save B matrices for training
    'use_localization': False,  # Use FULL B matrix (no localization)
    'inflate': [1.05, 0.0],  # [prior_inflation, relaxation]
    'F': 8.0,               # Lorenz 96 forcing
    'dt': 0.01}             # Time step

DA_exp_L20_EAKF = {
    'N_truth': 40,
    'N_DA': 40,
    'nens': 20,
    'DA_method': 'EAKF',
    'obs_freq': 4,          # Observe every 4 time steps
    'obs_err': 1.0,         # Observation error std
    'nobs': 20,             # Number of observations per cycle
    'loc_radius': 5.0,      # Localization radius
    'DA_freq': 4,           # DA cycle frequency
    'save_B': True,         # Save B matrices for training
    'use_localization': False,  # Use FULL B matrix (no localization)
    'inflate': [1.05, 0.0],  # [prior_inflation, relaxation]
    'F': 8.0,               # Lorenz 96 forcing
    'dt': 0.01}             # Time step

DA_exp_L20_ETKF = {
    'N_truth': 40,
    'N_DA': 40,
    'nens': 20,
    'DA_method': 'ETKF',
    'obs_freq': 4,          # Observe every 4 time steps
    'obs_err': 1.0,         # Observation error std
    'nobs': 20,             # Number of observations per cycle
    'loc_radius': 5.0,      # Localization radius
    'DA_freq': 4,           # DA cycle frequency
    'save_B': True,         # Save B matrices for training
    'use_localization': False,  # Use FULL B matrix (no localization)
    'inflate': [1.05, 0.0],  # [prior_inflation, relaxation]
    'F': 8.0,               # Lorenz 96 forcing
    'dt': 0.01}             # Time step

# Select experiments to run
# experiments = [DA_exp_L20,DA_exp_L20_w16,DA_exp_L20_w4]
# STEPS_TRAIN = [1000,1000, 1000]  # Number of time steps for training
# exp_name = ['L20','L20_w16', 'L20_w4']
# widths = [40,16, 4]

experiments = [DA_exp_L20_ETKF]
STEPS_TRAIN = [1000]
exp_name = ['L20_ETKF']
widths = [4]

if __name__ == '__main__':

    for i in range(len(experiments)):
        print('------------------')
        print(f'Running Experiment {exp_name[i]}')
        print('------------------')

        # Setup experiment
        DA_train = L96.L96_DA_exp(exp_id=exp_name[i], **experiments[i])
        
        # Construct expected filenames
        mean_folder = (f'EnsMean_{experiments[i]["DA_method"]}_N{experiments[i]["N_DA"]}_ens{experiments[i]["nens"]}_'
                      f'freq{experiments[i]["DA_freq"]}_relax{experiments[i]["inflate"][1]:.2f}_'
                      f'loc{experiments[i]["loc_radius"]:.1f}_nobs{experiments[i]["nobs"]}_'
                      f'err{experiments[i]["obs_err"]:.1e}')
        mean_file_path = f'{BASE_DIR}/{exp_name[i]}/{mean_folder}.nc'

        # Phase 1: Generate Training Data via EnKF
        if not os.path.exists(mean_file_path):
            print()
            print('>>>>Phase 1: Generate Training Data via Kalman Filter')
            
            # Generate truth trajectory
            DA_train.generate_truth(steps=2000, save_every=1)
            # DA_train.plot_truth(cut=500, steps = 2000)
            # break;
            
            # Generate observations
            DA_train.generate_obs(steps=2000, save_netcdf=True)
            
            # Spin up ensemble
            DA_train.ens_spinup(steps=100, save_netcdf=True)
            
            # Run EnKF experiment with save_B=True to collect B matrices
            DA_train.run_exp(DA_steps=STEPS_TRAIN[i], DA_start=0, ic_seed=0)
        else:
            print()
            print('>>>>Phase 1: SKIPPING (Data already exists)')
        ####note to self, add a check that the EnKF is working as intended (should be fine, but add for debugging purposes later)    

        print()
        print('>>>>Phase 2: Preprocess Data to Train UNet')
        
        # Construct B matrix folder name
        b_folder = (f'{experiments[i]["DA_method"]}_N{experiments[i]["N_DA"]}_ens{experiments[i]["nens"]}_'
                   f'freq{experiments[i]["DA_freq"]}_relax{experiments[i]["inflate"][1]:.2f}_'
                   f'loc{experiments[i]["loc_radius"]:.1f}_nobs{experiments[i]["nobs"]}_'
                   f'err{experiments[i]["obs_err"]:.1e}')
        
        # Load B matrices and ensemble mean
        B_ens_ds = xr.open_mfdataset(
            f'{BASE_DIR}/{exp_name[i]}/{b_folder}/B_ens_cycle*.nc',
            combine='nested',
            concat_dim='cycle'
        )
        mean_ds = xr.open_dataset(
            f'{BASE_DIR}/{exp_name[i]}/{mean_folder}.nc'
        ).load()


        # Split data: use second half for training (after spin-up)
        n_samples = len(B_ens_ds.cycle)
        spin_up = int(n_samples * 0.5)
        
        # Extract B matrices and state data
        B_ens = B_ens_ds.B_ens.isel(cycle=slice(spin_up, None))
        x_data = mean_ds.x.isel(time=slice(spin_up, None))
        
        # Compute normalization statistics
        B_std = B_ens.std().values
        x_std = x_data.std().values
        
        # Create normalization dataset
        ml_std_ds = xr.Dataset({
            'B_std': xr.DataArray(B_std),
            'x_std': xr.DataArray(x_std)
        })
        ml_std_ds.to_netcdf(f'{BASE_DIR}/{exp_name[i]}/std.nc')
        
        # Normalize data for training
        # Key point from paper: normalize by standard deviation, not by range
        B_data = (B_ens.values / B_std).astype(np.float32)
        x_norm = (x_data.values / x_std).astype(np.float32)
        
        # Reshape for UNet
        # For L96: treat as 1D "images" but need proper shape for CNN
        n_train_samples = B_data.shape[0]
        N = experiments[i]['N_DA']
        
        # Key difference from paper: The paper uses LOCALIZED covariances
        # For L96, we'll use full covariances but could also implement localized version
        
        # Reshape x: (samples, 1, N) for 1D convolutions
        x_unet = x_norm[:, np.newaxis, :]
        
        # Reshape B: For full matrix, we have (samples, N, N)
        # B_unet = B_data[:, newaxis, :, :]
        B_unet = B_data.astype(np.float32)
        
        print(f"x_unet shape: {x_unet.shape}")
        print(f"B_unet shape: {B_unet.shape}")
        print(f"B_ens mean: {np.mean(np.abs(B_ens))}")
        print(f"B_data mean: {np.mean(np.abs(B_data))}")
        print(f"x_data mean: {np.mean(np.abs(x_norm))}")
        print(f"B_data Max: {np.max(B_data)}")
        print(f"B_data Min: {np.min(B_data)}")
        print(f"B_data Std: {np.std(B_data)}")
        
        print()
        
        # Phase 3: Train UNet
        start_epoch = 0
        n_epochs = 100  # Paper uses ~200 epochs
        
        # Define UNet architecture following the paper
        # Paper uses U-Net with encoder-decoder structure
        class L96_UNet(torch.nn.Module):
            """
            U-Net for predicting full B matrix from L96 state
            Simplified 1D version of the paper's 2D U-Net
            """
            def __init__(self, channels_in=1, channels_out=2, width=16):
                super(L96_UNet, self).__init__()
                self.N = N
                
                ###UNet structure from paper
                # Encoder (Contracting Path)
                self.enc1 = self.conv_block(channels_in, width) 
                self.pool1 = nn.MaxPool1d(kernel_size = 2, stride = 2)
                self.enc2 = self.conv_block(width, width*2) 
                self.pool2 = nn.MaxPool1d(kernel_size = 2, stride = 2)

                # Bottleneck
                self.bottleneck = self.conv_block(width*2, width*4) 

                # Decoder (Expanding Path)
                self.up2 = nn.ConvTranspose1d(width*4, width*2, kernel_size=2, stride=2)
                self.dec2 = self.conv_block(width*4, width*2)
                self.up1 = nn.ConvTranspose1d(width*2, width, kernel_size=2, stride=2)
                self.dec1 = self.conv_block(width*2, width)

                # Final 1x1 Conv to match output channels
                # self.final = nn.Conv1d(width, width, kernel_size=1)
                self.final = nn.Linear(width*N, N*N)

            def conv_block(self, in_ch, out_ch):

            	return nn.Sequential(
            		nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            		nn.ReLU(inplace=True),
            		nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            		nn.ReLU(inplace=True)
            		)
                
            def forward(self, x):

                # Encoder
                e1 = self.enc1(x)
                p1 = self.pool1(e1)

                e2 = self.enc2(p1)
                p2 = self.pool1(e2)

                # Bottleneck
                b = self.bottleneck(p2)

                # Decoder with Skip Connections
                d2 = self.up2(b)
                d2 = torch.cat((d2, e2), dim=1) # Concatenation
                d2 = self.dec2(d2)

                d1 = self.up1(d2)
                d1 = torch.cat((d1, e1), dim=1) # Concatenation
                d1 = self.dec1(d1)

                # out = self.final(d1) ###old return statement

                # out = out[:,:,:self.N]

                # return out.view(x.size(0), 1, self.N, self.N)

                batch_size = d1.shape[0]
                d1_flat = d1.view(batch_size,-1) #make it (batch, width*N)
                B_flat = self.final(d1_flat) #make it (batch, N*N)

                B_pred = B_flat.view(batch_size, self.N, self.N) #make it (batch, N, N)

                return B_pred


        model = L96_UNet(width = widths[i])
        model = model.to(DEVICE)
        model = model.float()
        
        train_file = f'{BASE_DIR}/{exp_name[i]}/unet_epoch{n_epochs}.pt'
        
        if not os.path.exists(train_file):
            print('>>>>Phase 3: Training UNet')
            
            # Split into train/validation
            n_total = x_unet.shape[0]
            print('total samples: ', n_total)
            n_train_split = int(n_total * 0.8)
            print('number of training samples: ', n_train_split)
            
            # Convert to PyTorch tensors
            x_train = torch.from_numpy(x_unet[:n_train_split]).float()
            B_train = torch.from_numpy(B_unet[:n_train_split]).float()
            x_valid = torch.from_numpy(x_unet[n_train_split:]).float()
            B_valid = torch.from_numpy(B_unet[n_train_split:]).float()
            
            train_ds = torch.utils.data.TensorDataset(x_train, B_train)
            valid_ds = torch.utils.data.TensorDataset(x_valid, B_valid)
            
            params = {'batch_size': 128, 'num_workers': 2, 'shuffle': True}
            training_generator = torch.utils.data.DataLoader(train_ds, **params)
            validation_generator = torch.utils.data.DataLoader(valid_ds, **params)
            
            # Paper uses MSE loss and Adam optimizer with lr=0.002
            criterion = torch.nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.002)
            
            train_losses = []
            valid_losses = []
            best_valid_loss = float('inf')
            best_epoch = 0
            
            print(f'Training for {n_epochs} epochs...')
            
            for epoch in range(1, n_epochs + 1):
                # Training
                model.train()
                train_loss = 0.0
                for x_batch, B_batch in training_generator:
                    x_batch = x_batch.to(DEVICE)
                    B_batch = B_batch.to(DEVICE)
                    
                    optimizer.zero_grad()
                    B_pred = model(x_batch)
                    loss = criterion(B_pred, B_batch)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(training_generator)
                train_losses.append(train_loss)
                
                # Validation
                model.eval()
                valid_loss = 0.0
                with torch.no_grad():
                    for x_batch, B_batch in validation_generator:
                        x_batch = x_batch.to(DEVICE)
                        B_batch = B_batch.to(DEVICE)
                        
                        B_pred = model(x_batch)
                        loss = criterion(B_pred, B_batch)
                        valid_loss += loss.item()
                
                valid_loss /= len(validation_generator)
                valid_losses.append(valid_loss)

                # Track best model based on validation loss (following paper)
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_epoch = epoch
                    # Save best model
                    torch.save(model.state_dict(), 
                             f'{BASE_DIR}/{exp_name[i]}/unet_best.pt')
                
                if epoch % 10 == 0:
                    print(f'Epoch {epoch}/{n_epochs}, Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}')
                
                # Save checkpoint every 10 epochs
                if epoch % 10 == 0:
                    torch.save(model.state_dict(), 
                             f'{BASE_DIR}/{exp_name[i]}/unet_epoch{epoch}.pt')
            
            print(f'Best validation loss: {best_valid_loss:.6f} at epoch {best_epoch}')
            print(f'Following paper: Use model from epoch {best_epoch} for UNetKF')
            
            #save training and validation loss for experiment for plotting later
            with open(f"{BASE_DIR}/{exp_name[i]}/losses.csv", "w", newline="") as f:
                writer = csv.writer(f)
                # Optional: header
                writer.writerow(["train_loss", "valid_loss"])
    
                # Write rows for each epoch
                for t, v in zip(train_losses, valid_losses):
                    writer.writerow([t, v])
            print('Training and Validation Losses saved')
        else:
            print('>>>>Phase 3: SKIPPING (UNet already trained)')
        
        print()
        print('>>>>Phase 4: Running UNetKF')
        
        # Load best model (following paper's approach)
        model_file = f'{BASE_DIR}/{exp_name[i]}/unet_best.pt'
        if not os.path.exists(model_file):
            # Fallback to last epoch if best not saved
            model_file = f'{BASE_DIR}/{exp_name[i]}/unet_epoch{n_epochs}.pt'
        
        model.load_state_dict(torch.load(model_file, map_location=DEVICE))
        model.eval()
        ml_std_ds = xr.open_dataset(f'{BASE_DIR}/{exp_name[i]}/std.nc')
        
        # Setup UNetKF experiment
        # Key from paper: UNetKF typically uses single member (nens=1)
        # Paper also tests with small ensembles (5, 10, 20)
        DA_paras = {
            'N_truth': DA_train.N_truth,
            'N_DA': DA_train.N_DA,
            'nens': 1,  # Single member as in paper
            'DA_method': 'UnetKF',
            'obs_freq': DA_train.obs_freq,
            'obs_err': DA_train.obs_err,
            'nobs': DA_train.nobs,
            'loc_radius': DA_train.loc_radius,
            'DA_freq': DA_train.DA_freq,
            'save_B': False,
            'use_localization': False,  # Use full predicted B matrix
            'inflate': [1.05, 0.0],  # Paper uses no inflation for single member
            'training_exp': DA_train,
            'F': DA_train.F,
            'dt': DA_train.dt
        }
        
        DA_unet = L96.L96_DA_exp(exp_id=exp_name[i], **DA_paras)
        
        # Run UNetKF experiment
        output_str = f'/{exp_name[i]}_UnetKF'
        DA_unet.run_exp(
            DA_steps=1000,  # Test for 10000 steps
            DA_start=0,
            ic_seed=0,
            output_str=output_str,
            ml_model=model,
            ml_std_ds=ml_std_ds
        )
        
        print()
        print(f'Experiment {exp_name[i]} completed!')
        print('------------------')