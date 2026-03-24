import numpy as np
import xarray as xr
from numpy.random import default_rng
from os.path import exists
from pathlib import Path
from numba import jit
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from models import L96Model, QG1Model, Ensemble

# ─────────────────────────────────────────────────────────────────────────────
# Base data directory — override per experiment via exp_id
# ─────────────────────────────────────────────────────────────────────────────
data_dir = './data'


# =============================================================================
# BASE CLASS — model-agnostic DA logic
# =============================================================================

class DA_Experiment:
    """
    Base class for data assimilation experiments.

    Contains all logic that is independent of the underlying dynamical model:
    observation generation, the DA update step, running experiments, and saving
    results.  Model-specific behaviour (spinup, truth generation, diagnostics,
    ensemble initialisation) is implemented in subclasses.

    Subclasses must implement:
        - ens_spinup(steps, ...)
        - generate_truth(steps, ...)
        - init_DA(steps, ic_seed)
        - file_name()
        - (optionally) diagnose_truth(steps)
    """

    def __init__(self, N=40, **kwargs):
        self.N_truth = N
        self.N_DA    = kwargs.get('N_DA', N)
        self.nobs    = kwargs.get('nobs', 20)
        self.obs_freq  = kwargs.get('obs_freq', 5)
        self.obs_err   = kwargs.get('obs_err', 1.0)
        self.obs_type  = kwargs.get('obs_type', 'regular')
        self.DA_method = kwargs.get('DA_method', 'NoDA')
        self.nens      = kwargs.get('nens', 1)
        self.F         = kwargs.get('F', 8.0)
        self.dt        = kwargs.get('dt', 0.01)

        if self.DA_method in ['EnKF', 'ETKF', 'EAKF', 'UNetKF']:
            self.inflate = kwargs.get('inflate', 1.0)
            self.save_B  = kwargs.get('save_B', False)

        self.exp_id   = kwargs.get('exp_id', 'default')
        self.model_id = kwargs.get('model_id', 'model')
        self.save_dir = str(Path(data_dir) / self.model_id / self.exp_id)
        Path(self.save_dir).mkdir(parents = True, exist_ok = True)
        self.read_dir  = self.save_dir
        truth_run_dir = str(Path(data_dir) / self.model_id)
        Path(truth_run_dir).mkdir(parents = True, exist_ok = True)
        self.truth_dir = truth_run_dir  # shared across experiments with same F and dt

    # -------------------------------------------------------------------------
    # Observation methods
    # -------------------------------------------------------------------------

    def generate_obs(self, steps, save_netcdf=True, overwrite=False):
        """Sample observations from the truth run at the experiment obs frequency."""

        truth_ds = self.read_truth(steps=steps)

        obs_step_indices = range(self.obs_freq - 1, len(truth_ds.time), self.obs_freq)
        obs_times  = truth_ds.time[obs_step_indices]
        n_time     = len(obs_times)
        x_truth    = truth_ds.x.isel(time=obs_step_indices)

        obs_x       = np.zeros((n_time, self.nobs))
        obs_idx     = np.zeros((n_time, self.nobs), dtype=int)
        obs_err_std = np.ones((n_time, self.nobs)) * self.obs_err

        rng = default_rng()
        P = int(self.N_truth / self.nobs)   # chunk size for LHS

        # Latin hypercube indices drawn once (reused for random_once)
        ran_idx_once = np.array([rng.integers(k * P, (k + 1) * P) for k in range(self.nobs)])

        for t in range(n_time):
            if self.obs_type == 'random_once':
                idx = ran_idx_once
            elif self.obs_type == 'random_every':
                idx = np.array([rng.integers(k * P, (k + 1) * P) for k in range(self.nobs)])
            else:
                idx = np.arange(0, self.N_truth, P)

            obs_idx[t, :] = idx
            obs_x[t, :]   = x_truth.values[t, idx] + rng.standard_normal(self.nobs) * self.obs_err

        obs_ds = xr.Dataset(
            {
                'x':       (['cycle', 'obs'], obs_x),
                'idx':     (['cycle', 'obs'], obs_idx),
                'err_std': (['cycle', 'obs'], obs_err_std),
                'time':    (['cycle'], obs_times.values),
            },
            coords={'cycle': range(n_time), 'obs': range(self.nobs)},
            attrs={'nobs': self.nobs, 'obs_error': self.obs_err, 'obs_freq': self.obs_freq},
        )

        if save_netcdf:
            file_name = (f'{self.save_dir}/Obs_N{self.N_truth}_freq{self.obs_freq}'
                         f'_nobs{self.nobs}_err{self.obs_err:.1e}.nc')
            if not exists(file_name) or overwrite:
                obs_ds.to_netcdf(file_name)

        return obs_ds

    def read_obs(self, folder=''):
        """Read observation file from disk."""
        if folder:
            obs_file = (f'{self.read_dir}/{folder}/Obs_N{self.N_truth}_freq{self.obs_freq}'
                        f'_nobs{self.nobs}_err{self.obs_err:.1e}.nc')
        else:
            obs_file = (f'{self.read_dir}/Obs_N{self.N_truth}_freq{self.obs_freq}'
                        f'_nobs{self.nobs}_err{self.obs_err:.1e}.nc')
        return xr.open_dataset(obs_file)

    def read_truth(self, steps, folder=''):
        """Read truth file from disk. Subclasses may override if naming differs."""
        raise NotImplementedError("read_truth must be implemented by the subclass.")

    def create_obs_operator(self, obs_cycle_idx):
        """Build H (observation operator) and R (obs error covariance) for one cycle."""
        nobs = self.nobs
        N    = self.N_DA

        obs_idx = self.obs_ds.idx[obs_cycle_idx].values
        obs_err = self.obs_ds.err_std[obs_cycle_idx].values

        H = np.zeros((nobs, N))
        H[range(nobs), obs_idx] = 1.0
        R = np.diag(obs_err ** 2)

        return H, R

    # -------------------------------------------------------------------------
    # DA update step
    # -------------------------------------------------------------------------

    def assimilation(self, forecast, obs_cycle_idx, **kwargs):
        """Apply the specified DA method to produce a posterior ensemble."""

        if isinstance(forecast, xr.DataArray):
            prior = forecast[-1, :, :].values
        else:
            prior = forecast[-1, :, :]

        H, R_obs = self.create_obs_operator(obs_cycle_idx)
        obs_x    = self.obs_ds.x[obs_cycle_idx].values
        B_out    = None

        if self.DA_method in ['EnKF', 'EAKF', 'ETKF']:
            if self.DA_method == 'ETKF':
                posterior, B_out = ETKF(prior, obs_x, H, R_obs, self.inflate)
            elif self.DA_method == 'EAKF':
                posterior, B_out = EAKF(prior, obs_x, H, R_obs, self.inflate)
            else:
                prior  = ens_inflate(prior, self.inflate)
                B_out  = calculate_cov(prior)
                posterior = EnKF(prior, obs_x, H, R_obs, B_out)

            if not self.save_B:
                B_out = None

        elif self.DA_method == 'UNetKF':
            device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            ml_model = kwargs['ml_model'].to(device)
            ml_std   = kwargs['std_file']

            prior_mean = prior.mean(axis=0) if self.nens > 1 else prior[0]
            x_input    = prior_mean / ml_std.x_std.values

            x_tensor = torch.from_numpy(x_input[np.newaxis, np.newaxis, :]).float().to(device)
            B_pred   = ml_model(x_tensor).to('cpu').detach().numpy()[0, :, :]
            B_pred   = (B_pred * ml_std.B_std.values).astype(np.float64)

            if self.save_B:
                B_out = B_pred

            if self.nens == 1:
                posterior = EnKF_mean(prior[0], obs_x, H, R_obs, B_pred)
                posterior = posterior[np.newaxis, :]
            else:
                posterior = EnKF(prior, obs_x, H, R_obs, B_pred)

        else:
            posterior = prior

        return posterior, B_out

    # -------------------------------------------------------------------------
    # Experiment runner
    # -------------------------------------------------------------------------

    def run_exp(self, DA_steps, truth_steps, ic_seed, output_str='', **kwargs):
        """Run a full DA experiment and save results."""

        self.init_DA(truth_steps, ic_seed=ic_seed)

        DA_kwargs = {}
        if self.DA_method == 'UNetKF':
            DA_kwargs['ml_model'] = kwargs['ml_model']
            DA_kwargs['std_file'] = kwargs['ml_std_ds']

        n_cycles         = DA_steps // self.obs_freq
        forecast_results = []
        analysis_results = []
        B_list           = []

        for cycle in tqdm(range(n_cycles), desc=f"Running {self.DA_method}"):

            forecast = self.ens.run_for_steps(self.obs_freq, save_every=self.obs_freq)

            if isinstance(forecast, xr.DataArray):
                forecast_state = forecast[-1, :, :].values
            else:
                forecast_state = forecast[-1, :, :]
            forecast_results.append(forecast_state)

            if self.DA_method != 'NoDA':
                analysis, B_out = self.assimilation(forecast, cycle, **DA_kwargs)
                if B_out is not None:
                    B_list.append(B_out)
                for i, model in enumerate(self.ens.models):
                    model.x = analysis[i]
                analysis_results.append(analysis)
            else:
                analysis_results.append(forecast_state)

        # Save covariance matrices if requested
        if self.save_B and len(B_list) > 0:
            B_all = np.stack(B_list, axis=0)
            xr.Dataset({
                'B_ens': xr.DataArray(B_all, dims=['cycle', 'dim', 'dim_d'],
                                      coords={'cycle': range(len(B_list))})
            }).to_netcdf(f'{self.save_dir}/B_ens_{self.file_name()}.nc')

        # Package and save results
        forecast_array = np.array(forecast_results)
        analysis_array = np.array(analysis_results)
        times = np.arange(n_cycles) * self.obs_freq * self.dt

        forecast_xr = xr.DataArray(forecast_array, dims=['time', 'model', 'dim'],
                                   coords={'time': times, 'model': range(self.nens),
                                           'dim': range(self.N_DA)})
        analysis_xr = xr.DataArray(analysis_array, dims=['time', 'model', 'dim'],
                                   coords={'time': times, 'model': range(self.nens),
                                           'dim': range(self.N_DA)})

        forecast_mean = forecast_xr.mean(dim='model')
        analysis_mean = analysis_xr.mean(dim='model')

        target_dir = f'{self.save_dir}{output_str}'
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        file_name = self.file_name()

        xr.Dataset({'x': analysis_mean}).to_netcdf(f'{target_dir}/EnsMean_{file_name}.nc', mode='w')

        if self.nens > 1:
            xr.Dataset({'x': analysis_xr.std(dim='model')}).to_netcdf(
                f'{target_dir}/EnsStd_{file_name}.nc', mode='w')
            xr.Dataset({'x': forecast_mean}).to_netcdf(
                f'{target_dir}/ForecastMean_{file_name}.nc', mode='w')
            xr.Dataset({'x': forecast_xr.std(dim='model')}).to_netcdf(
                f'{target_dir}/ForecastStd_{file_name}.nc', mode='w')

    # -------------------------------------------------------------------------
    # Abstract interface — subclasses must implement these
    # -------------------------------------------------------------------------

    def ens_spinup(self, steps, **kwargs):
        raise NotImplementedError("ens_spinup must be implemented by the subclass.")

    def generate_truth(self, steps, spin_up):
        raise NotImplementedError("generate_truth must be implemented by the subclass.")

    def init_DA(self, steps, ic_seed=0):
        raise NotImplementedError("init_DA must be implemented by the subclass.")

    def file_name(self):
        raise NotImplementedError("file_name must be implemented by the subclass.")


# =============================================================================
# L96 SUBCLASS
# =============================================================================

class L96_DA_Experiment(DA_Experiment):
    """
    DA experiment subclass for the Lorenz 96 model.

    Handles L96-specific spinup, truth generation, diagnostics,
    and ensemble initialisation.
    """

    def __init__(self, N=40, **kwargs):
        super().__init__(N=N, **kwargs)

    # -------------------------------------------------------------------------
    # Spinup
    # -------------------------------------------------------------------------

    def ens_spinup(self, steps, save_netcdf=True, overwrite=False):
        """Spin up ensemble members to ensure they are on the L96 attractor."""

        rng = default_rng()
        ens = Ensemble([L96Model(N=self.N_truth, F=self.F, dt=self.dt)
                        for _ in range(self.nens)])

        for model in ens.models:
            pert = rng.standard_normal(self.N_truth) * 0.01
            model.x = (self.F * np.ones(self.N_truth) + pert).copy()

        for _ in range(steps):
            ens.step_forward()

        x_init = np.array([m.x for m in ens.models])
        x_init_da = xr.DataArray(
            x_init,
            dims=['model', 'dim'],
            coords={'model': range(self.nens), 'dim': range(self.N_truth)},
        )

        if save_netcdf:
            file_name = (f'{self.save_dir}/IC_x_N{self.N_truth}'
                         f'_ens{self.nens}_dt{self.dt}.nc')
            if not exists(file_name) or overwrite:
                x_init_da.to_netcdf(file_name)

        return x_init_da

    # -------------------------------------------------------------------------
    # Truth generation
    # -------------------------------------------------------------------------

    def generate_truth(self, steps, spin_up):
        """Generate and save a truth trajectory for L96."""

        ic_file = f'{self.read_dir}/IC_x_N{self.N_truth}_ens1_dt{self.dt}.nc'

        if exists(ic_file):
            x_init = xr.open_dataarray(ic_file)
        else:
            temp_exp = L96_DA_Experiment(N=self.N_truth, nens=1, F=self.F,
                                         dt=self.dt, exp_id=self.exp_id)
            x_init = temp_exp.ens_spinup(steps=spin_up)

        model   = L96Model(N=self.N_truth, F=self.F, dt=self.dt)
        model.x = x_init[0, :].values

        results = []
        times   = []

        for _ in tqdm(range(steps), desc="Generating L96 truth"):
            model.step_forward()
            results.append(model.x.copy())
            times.append(model.t)

        x_truth  = np.array(results)
        ds_truth = xr.Dataset(
            {'x': (['time', 'dim'], x_truth)},
            coords={'time': times, 'dim': range(self.N_truth)},
        )

        file_name = (f'{self.truth_dir}/Truth_N{self.N_truth}'
                     f'_F{self.F}_dt{self.dt}_{steps}steps.nc')
        if not exists(file_name):
            ds_truth.to_netcdf(file_name)

        return ds_truth

    def read_truth(self, steps, folder=''):
        """Read L96 truth file from disk."""
        base = f'{self.truth_dir}/{folder}' if folder else self.truth_dir
        truth_file = f'{base}/Truth_N{self.N_truth}_F{self.F}_dt{self.dt}_{steps}steps.nc'
        truth_ds = xr.open_dataset(truth_file)
        truth_ds.attrs['truth_file'] = truth_file
        return truth_ds

    # -------------------------------------------------------------------------
    # DA initialisation
    # -------------------------------------------------------------------------

    def init_DA(self, steps, ic_seed=0):
        """
        Initialise DA ensemble near the true initial state with climatological spread.
        Follows Majda and Harlim (2012) Chapter 11.
        """

        self.ens     = Ensemble([L96Model(N=self.N_DA, F=self.F, dt=self.dt)
                                 for _ in range(self.nens)])
        self.obs_ds  = self.read_obs()

        truth_ds    = self.read_truth(steps=steps)
        x_truth_t0  = truth_ds['x'].values[0]
        clim_std    = truth_ds['x'].values.std(axis=0).mean()

        rng_ic = default_rng(ic_seed)
        for model in self.ens.models:
            model.x = x_truth_t0 + rng_ic.standard_normal(self.N_DA) * clim_std

        return self.ens

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def diagnose_truth(self, steps):
        """Print summary statistics and plots for the L96 truth run."""

        truth_ds = self.read_truth(steps=steps)
        x = truth_ds['x'].values

        clim_mean     = x.mean()
        clim_var      = x.var(axis=0).mean()
        clim_std      = x.std(axis=0).mean()
        clim_var_time = x.var(axis=0)

        print("=== L96 Truth Diagnostics ===")
        print(f"  Steps:                {x.shape[0]}")
        print(f"  N variables:          {x.shape[1]}")
        print(f"  Climatological mean:  {clim_mean:.3f}")
        print(f"  Climatological var:   {clim_var:.3f}")
        print(f"  Climatological std:   {clim_std:.3f}")
        print(f"  Min var per variable: {clim_var_time.min():.3f}")
        print(f"  Max var per variable: {clim_var_time.max():.3f}")

        lyap, _ = self.estimate_lyapunov()
        print(f"\n  Lyapunov exponent:    {lyap:.3f}")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].bar(range(x.shape[1]), clim_var_time)
        axes[0].set_xlabel('Variable index')
        axes[0].set_ylabel('Variance')
        axes[0].set_title('Per-variable climatological variance')

        t = truth_ds['time'].values
        for i in [0, 10, 20]:
            axes[1].plot(t[:500], x[:500, i], label=f'x[{i}]', alpha=0.7)
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('State')
        axes[1].set_title('Truth trajectory (first 500 steps)')
        axes[1].legend()

        axes[2].hist(x.ravel(), bins=100, density=True)
        axes[2].set_xlabel('State value')
        axes[2].set_ylabel('Density')
        axes[2].set_title(f'State distribution (mean={clim_mean:.2f}, std={clim_std:.2f})')

        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/truth_diagnostics.png', dpi=150)

        return {
            'clim_mean': clim_mean,
            'clim_var':  clim_var,
            'clim_std':  clim_std,
            'lyapunov':  lyap,
            'lyap_time': 1.0 / lyap,
        }

    def estimate_lyapunov(self, steps=5000, perturbation=1e-8):
        """Estimate the leading Lyapunov exponent of the L96 system."""

        print(f"dt = {self.dt}")
        print(f"obs_freq = {self.obs_freq}")
        print(f"T_obs    = {self.obs_freq * self.dt:.4f}")

        m1 = L96Model(N=self.N_truth, F=self.F, dt=self.dt)
        m2 = L96Model(N=self.N_truth, F=self.F, dt=self.dt)
        m1.x = self.F + np.random.randn(self.N_truth) * 0.1

        for _ in range(10000):
            m1.step_forward()

        m2.x = m1.x + perturbation
        lyap_sum = 0.0
        m1_xs    = []

        for _ in range(steps):
            m1.step_forward()
            m2.step_forward()
            m1_xs.append(m1.x)
            dist      = np.linalg.norm(m1.x - m2.x)
            lyap_sum += np.log(dist / perturbation)
            m2.x      = m1.x + perturbation * (m2.x - m1.x) / dist

        lyap = lyap_sum / (steps * self.dt)
        print(f"final lyap = {lyap:.4f}")
        return lyap, m1_xs

    # -------------------------------------------------------------------------
    # File naming
    # -------------------------------------------------------------------------

    def file_name(self):
        if self.DA_method == 'NoDA':
            return f'Control_N{self.N_DA}'
        elif self.DA_method in ['EnKF', 'ETKF', 'EAKF']:
            return (f'{self.DA_method}_N{self.N_DA}_ens{self.nens}_freq{self.obs_freq}'
                    f'_nobs{self.nobs}_err{self.obs_err:.1e}')
        elif self.DA_method == 'UNetKF':
            return (f'UNetKF_N{self.N_DA}_ens{self.nens}_freq{self.obs_freq}'
                    f'_nobs{self.nobs}_err{self.obs_err:.1e}')
        else:
            return f'{self.DA_method}_N{self.N_DA}'


# =============================================================================
# QG SUBCLASS STUB
# =============================================================================

class QG1_DA_Experiment(DA_Experiment):
    """
    DA experiment subclass for a 1-layer quasi-geostrophic model.

    This is a stub. To activate it:
        1. Implement QGModel in models.py (see its docstring for guidance).
        2. Fill in ens_spinup, generate_truth, init_DA, and read_truth below,
           following the same pattern as L96_DA_Experiment.
        3. Adjust file_name() to reflect QG-specific parameters (e.g. nx, ny).

    Key differences from L96 to keep in mind:
        - The state vector is a flattened 2D field, so N = nx * ny.
        - Observation operators may need to account for the 2D grid structure.
        - Spinup times and perturbation scales will differ from L96.
        - If using pyqg, the model timestep and units differ from L96.
    """

    def __init__(self, nx=64, ny=64, **kwargs):
        self.nx = nx
        self.ny = ny
        N = nx * ny
        super().__init__(N=N, **kwargs)

    def ens_spinup(self, steps, save_netcdf=True, overwrite=False):
        raise NotImplementedError("QG1 ens_spinup not yet implemented.")

    def generate_truth(self, steps, spin_up):
        raise NotImplementedError("QG1 generate_truth not yet implemented.")

    def read_truth(self, steps, folder=''):
        raise NotImplementedError("QG1 read_truth not yet implemented.")

    def init_DA(self, steps, ic_seed=0):
        raise NotImplementedError("QG1 init_DA not yet implemented.")

    def file_name(self):
        raise NotImplementedError("QG1 file_name not yet implemented.")


# =============================================================================
# U-Net model (predicts full covariance matrix)
# =============================================================================

class DA_UNet(torch.nn.Module): 
    def __init__(self, N=40, width=16, dropout=0.0, spatial_dims=1, in_channels=1):
        super(DA_UNet, self).__init__()
        self.N           = N
        self.spatial_dims = spatial_dims
 
        # Select conv primitives based on spatial_dims
        if spatial_dims == 1:
            Conv        = nn.Conv1d
            ConvT       = nn.ConvTranspose1d
            MaxPool     = nn.MaxPool1d
            Dropout     = nn.Dropout1d
        elif spatial_dims == 2:
            Conv        = nn.Conv2d
            ConvT       = nn.ConvTranspose2d
            MaxPool     = nn.MaxPool2d
            Dropout     = nn.Dropout2d
        else:
            raise ValueError(f"spatial_dims must be 1 or 2, got {spatial_dims}")
 
        self._Conv    = Conv
        self._ConvT   = ConvT
        self._Dropout = Dropout
 
        self.enc1       = self._conv_block(in_channels, width)
        self.pool1      = MaxPool(kernel_size=2, stride=2)
        self.enc2       = self._conv_block(width, width * 2)
        self.pool2      = MaxPool(kernel_size=2, stride=2)
        self.bottleneck = self._conv_block(width * 2, width * 4)
        self.up2        = ConvT(width * 4, width * 2, kernel_size=2, stride=2)
        self.dec2       = self._conv_block(width * 4, width * 2, dropout=dropout)
        self.up1        = ConvT(width * 2, width, kernel_size=2, stride=2)
        self.dec1       = self._conv_block(width * 2, width, dropout=dropout)
        self.final      = nn.Linear(width * N, N * N)
 
    def _conv_block(self, in_ch, out_ch, dropout=0.0):
        layers = [
            self._Conv(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(self._Dropout(p=dropout))
        layers += [
            self._Conv(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)
 
    def forward(self, x):
        e1 = self.enc1(x);             p1 = self.pool1(e1)
        e2 = self.enc2(p1);            p2 = self.pool2(e2)
        b  = self.bottleneck(p2)
        d2 = self.dec2(torch.cat([self.up2(b),  e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1.reshape(x.size(0), -1)).view(x.size(0), self.N, self.N)

# =============================================================================
# DA functions
# =============================================================================

@jit(nopython=True)
def calculate_cov(data):
    return np.cov(data.T)


def EnKF(prior, obs, H, R, B):
    """Stochastic Ensemble Kalman Filter update."""

    nens, N = prior.shape
    nobs    = obs.shape[0]
    D       = H @ B @ H.T + R
    K       = np.linalg.solve(D.T, (B @ H.T).T).T

    obs_noise_std = np.sqrt(np.diag(R))
    noise = np.random.standard_normal((nobs, nens))
    for j in range(nobs):
        noise[j, :] *= obs_noise_std[j]

    obs_ens    = np.tile(obs[:, np.newaxis], (1, nens)) + noise
    innovation = obs_ens - (H @ prior.T)
    return (prior.T + K @ innovation).T


@jit(nopython=True)
def EnKF_mean(prior, obs, H, R, B):
    """Deterministic EnKF update for a single ensemble member."""
    D = H @ B @ H.T + R
    K = B @ H.T @ np.linalg.inv(D)
    return prior + K @ (obs - H @ prior)


def ETKF(prior, obs, H, R, inflate=1.0):
    """
    Ensemble Transform Kalman Filter.
    Majda and Harlim (2012), Chapter 9.
    """

    nens, N = prior.shape

    # Step 1 — prior mean and anomalies
    Xf_mean = prior.mean(axis=0)
    Xf      = prior - Xf_mean
    Yf      = Xf @ H.T
    Yf_mean = H @ Xf_mean

    # Step 2 — form K×K matrix J with inflation baked in
    R_inv = np.linalg.inv(R)
    J     = ((nens - 1) / inflate) * np.eye(nens) + Yf @ R_inv @ Yf.T
    X, gamma, Xt = np.linalg.svd(J)

    # Step 3 — Kalman gain
    K_gain = Xf.T @ (X * (1 / gamma) @ Xt) @ Yf @ R_inv

    # Step 4 — posterior mean
    Xa_mean = Xf_mean + K_gain @ (obs - Yf_mean)

    # Step 5 — transform matrix and posterior perturbations
    T              = np.sqrt(nens - 1) * (X * (1 / np.sqrt(gamma)) @ Xt)
    Xf_posterior   = Xf.T @ T

    # Step 6 — posterior ensemble
    posterior = Xa_mean + Xf_posterior.T
    R_prior   = calculate_cov(prior)

    return posterior, R_prior


def EAKF(prior, obs, H, R, inflate=1.0):
    """
    Ensemble Adjustment Kalman Filter.
    Majda and Harlim (2012), Chapter 9.
    """

    nens, N = prior.shape

    # Step 1 — prior mean and anomaly matrix
    Xf_mean = prior.mean(axis=0)
    Xf      = prior - Xf_mean

    # Step 2 — eigendecompose inflated prior covariance: F·Σ²·F^T
    R_prior = (inflate / (nens - 1)) * (Xf.T @ Xf)
    eigenvals_full, F_full = np.linalg.eigh(R_prior)

    n_trunc = nens - 1
    S2      = eigenvals_full[-n_trunc:]
    F_eig   = F_full[:, -n_trunc:]
    S       = np.sqrt(S2)

    # Step 3 — eigendecompose Σ·F^T·G^T·R^{-1}·G·F·Σ = X·D·X^T
    R_inv = np.linalg.inv(R)
    inner = (S[:, np.newaxis]
             * (F_eig.T @ H.T @ R_inv @ H @ F_eig)
             * S[np.newaxis, :])
    D_diag, X_eig = np.linalg.eigh(inner)

    # Step 4 — adjustment matrix A = F·Σ·X·(I+D)^{-1/2}·Σ^{-1}·F^T
    S_inv        = 1.0 / S
    IpD_inv_sqrt = np.diag(1.0 / np.sqrt(1.0 + D_diag))
    A_mat = (F_eig
             @ np.diag(S)
             @ X_eig
             @ IpD_inv_sqrt
             @ np.diag(S_inv)
             @ F_eig.T)
    Xa = Xf @ A_mat.T

    # Step 5 — solve L·ū_a = y for posterior mean
    FtS2inv_Ft = F_eig @ np.diag(S_inv ** 2) @ F_eig.T
    GtRinvG    = H.T @ R_inv @ H
    L = FtS2inv_Ft + GtRinvG
    y = FtS2inv_Ft @ Xf_mean + H.T @ R_inv @ obs
    Xa_mean = np.linalg.solve(L, y)

    # Step 6 — posterior ensemble
    posterior = Xa_mean + Xa

    return posterior, R_prior


@jit(nopython=True)
def ens_inflate(prior, factor):
    """Multiplicative covariance inflation applied to the prior ensemble."""
    nens, N     = prior.shape
    inflated    = np.zeros(prior.shape)
    mean_prior  = np.zeros(N)
    for j in range(N):
        mean_prior[j] = np.sum(prior[:, j]) / nens
    for i in range(nens):
        inflated[i, :] = mean_prior + factor * (prior[i, :] - mean_prior)
    return inflated