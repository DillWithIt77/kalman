import numpy as np
import xarray as xr
from numpy.random import default_rng
from os.path import exists
from pathlib import Path
from numba import jit
from tqdm import tqdm
import torch
from numpy.linalg import matrix_rank
import os
import matplotlib.pyplot as plt
from torch import optim
import logging
import torch.nn as nn
import torch.nn.functional as F

rng = default_rng()

# Configure your data directory
data_dir = './data/lorenz96'

class Lorenz96Model:
    """Lorenz 96 model implementation"""
    def __init__(self, N=40, F=8.0, dt=0.01):
        """
        N: number of variables (state dimension)
        F: forcing parameter
        dt: time step
        """
        self.N = N
        self.F = F
        self.dt = dt
        self.x = np.zeros(N)
        self.t = 0.0
        
    def rhs(self, x):
        """Right-hand side of Lorenz 96 equations"""
        N = self.N
        dxdt = np.zeros(N)
        for i in range(N):
            dxdt[i] = (x[(i+1)%N] - x[(i-2)%N]) * x[(i-1)%N] - x[i] + self.F
        return dxdt
    
    def step_forward(self):
        """4th order Runge-Kutta time step"""
        x = self.x
        dt = self.dt
        
        k1 = self.rhs(x)
        k2 = self.rhs(x + 0.5*dt*k1)
        k3 = self.rhs(x + 0.5*dt*k2)
        k4 = self.rhs(x + dt*k3)
        
        self.x = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        self.t += dt
        
    def to_dataset(self):
        """Convert current state to xarray Dataset"""
        return xr.Dataset({
            'x': (['dim'], self.x),
            't': self.t
        })

class L96_DA_exp:
    """Data Assimilation experiment framework for Lorenz 96"""
    
    def __init__(self, N_truth=40, **kwargs):
        """
        N_truth [int]: State dimension for "truth" model
        N_DA [int]: State dimension for DA model (can be same or different)
        obs_freq [int]: Observation frequency (time steps)
        obs_err [float]: Standard deviation for observation errors
        DA_method [str]: 'NoDA'/'3DVar'/'EnKF'
        DA_frequency [int]: DA cycle frequency (time steps)
        nens [int]: Ensemble size (1 for 3DVar)
        nobs [int]: Number of observations
        loc_radius [float]: Localization radius for covariance
        """
        self.N_truth = N_truth
        self.N_DA = kwargs.get('N_DA', N_truth)
        self.nobs = kwargs.get('nobs', 20)
        self.obs_freq = kwargs.get('obs_freq', 4)
        self.obs_err = kwargs.get('obs_err', 1.0)
        self.DA_method = kwargs.get('DA_method', 'NoDA')
        self.nens = kwargs.get('nens', 1)
        self.DA_freq = kwargs.get('DA_freq', 4)
        self.loc_radius = kwargs.get('loc_radius', 5.0)
        self.use_localization = kwargs.get('use_localization', True)
        self.F = kwargs.get('F', 8.0)
        self.dt = kwargs.get('dt', 0.01)
        
        if self.DA_method in ['EnKF']:
            self.delta_steps = kwargs.get('delta_steps', 100)
        
        if self.DA_method == 'EnKF':
            self.inflate = kwargs.get('inflate', [1.0, 0.0])
            self.save_B = kwargs.get('save_B', False)
        
        if self.DA_method == 'UnetKF':
            self.inflate = kwargs.get('inflate', [1.0, 0.0])
            self.save_B = kwargs.get('save_B', False)
            self.training_exp = kwargs.get('training_exp', None)
        
        # Setup directories
        self.exp_id = kwargs.get('exp_id', 'default')
        self.save_dir = str(Path(data_dir) / self.exp_id)
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        self.read_dir = self.save_dir
        
    def ens_spinup(self, steps=10000, save_netcdf=True, overwrite=False):
        """Spin up model ensemble"""
        ens = L96Ensemble([Lorenz96Model(N=self.N_truth, F=self.F, dt=self.dt) 
                          for i in range(self.nens)])

        # Initialize with small perturbations - Unique per member
        for i, model in enumerate(ens.models):
            pert = rng.standard_normal(self.N_truth) * 0.01
            model.x = (self.F * np.ones(self.N_truth) + pert).copy()

        # # Initialize with small perturbations
        # for model in ens.models:
        #     model.x = self.F * np.ones(self.N_truth) + rng.standard_normal(self.N_truth) * 0.01
        
        # Spin up
        for _ in range(steps):
            ens.step_forward()
        
        # Save initial conditions
        x_init = np.array([m.x for m in ens.models])
        x_init_da = xr.DataArray(x_init, dims=['model', 'dim'], 
                                 coords={'model': range(self.nens), 
                                        'dim': range(self.N_truth)})
        
        if save_netcdf:
            file_name = f'{self.save_dir}/IC_x_N{self.N_truth}_ens{self.nens}.nc'
            if not exists(file_name) or overwrite:
                x_init_da.to_netcdf(file_name)
        
        return x_init_da
    
    def generate_truth(self, steps, save_every=10):
        """Generate truth trajectory"""
        ic_file = f'{self.read_dir}/IC_x_N{self.N_truth}_ens1.nc'
        
        if exists(ic_file):
            x_init = xr.open_dataarray(ic_file)
        else:
            # Create single member ensemble for truth
            temp_exp = L96_DA_exp(N_truth=self.N_truth, nens=1, F=self.F, dt=self.dt, exp_id=self.exp_id)
            x_init = temp_exp.ens_spinup()
        
        model = Lorenz96Model(N=self.N_truth, F=self.F, dt=self.dt)
        model.x = x_init[0, :].values
        
        results = []
        times = []
        
        for i in tqdm(range(steps), desc="Generating truth"):
            model.step_forward()
            if (i + 1) % save_every == 0:
                results.append(model.x.copy())
                times.append(model.t)
        
        x_truth = np.array(results)
        ds_truth = xr.Dataset({
            'x': (['time', 'dim'], x_truth)
        }, coords={
            'time': times,
            'dim': range(self.N_truth)
        })
        
        file_name = f'{self.save_dir}/Truth_N{self.N_truth}_{steps}steps.nc'
        if not exists(file_name):
            ds_truth.to_netcdf(file_name)
        
        return ds_truth
    
    def generate_obs(self, steps=10000, save_netcdf=True, overwrite=False):
        """Sample observations from truth simulation"""
        truth_ds = self.read_truth(steps=steps)
        
        # Select observation times - align with DA cycles
        # If DA_freq=4, we want observations at steps [3, 7, 11, ...] which are indices [0, 1, 2, ...]
        # when we subsample every obs_freq steps starting from obs_freq-1
        obs_step_indices = range(self.obs_freq - 1, len(truth_ds.time), self.obs_freq)
        obs_times = truth_ds.time[obs_step_indices]
        n_time = len(obs_times)
        x_truth = truth_ds.x.isel(time=obs_step_indices)
        
        # Initialize observation arrays
        obs_x = np.zeros((n_time, self.nobs))
        obs_idx = np.zeros((n_time, self.nobs), dtype=int)
        obs_err_std = np.ones((n_time, self.nobs)) * self.obs_err
        
        # Generate random observations with errors
        for t in range(n_time):
            # Randomly sample observation locations (changes with time as in paper)
            idx = rng.choice(self.N_truth, size=self.nobs, replace=False)
            obs_idx[t, :] = idx
            
            # Add observation errors
            obs_x[t, :] = x_truth.values[t, idx] + rng.standard_normal(self.nobs) * self.obs_err
        
        obs_ds = xr.Dataset({
            "x": (["cycle", "obs"], obs_x),  # Changed from time_idx to cycle
            "idx": (["cycle", "obs"], obs_idx),
            'err_std': (["cycle", "obs"], obs_err_std),
            'time': (["cycle"], obs_times.values)
        }, coords={
            'cycle': range(n_time),  # This matches the DA cycle index
            'obs': range(self.nobs)
        }, attrs={
            'nobs': self.nobs,
            'obs_error': self.obs_err,
            'obs_freq': self.obs_freq
        })
        
        if save_netcdf:
            file_name = f'{self.save_dir}/Obs_N{self.N_truth}_freq{self.obs_freq}_nobs{self.nobs}_err{self.obs_err:.1e}.nc'
            if not exists(file_name) or overwrite:
                obs_ds.to_netcdf(file_name)
        
        return obs_ds
    
    def read_truth(self, steps, folder=''):
        """Read truth trajectory"""
        if folder:
            truth_file = f'{self.read_dir}/{folder}/Truth_N{self.N_truth}_{steps}steps.nc'
        else:
            truth_file = f'{self.read_dir}/Truth_N{self.N_truth}_{steps}steps.nc'
        
        truth_ds = xr.open_dataset(truth_file)
        truth_ds.attrs['truth_file'] = truth_file
        return truth_ds
    
    def read_obs(self, folder=''):
        """Read observations"""
        if folder:
            obs_file = f'{self.read_dir}/{folder}/Obs_N{self.N_truth}_freq{self.obs_freq}_nobs{self.nobs}_err{self.obs_err:.1e}.nc'
        else:
            obs_file = f'{self.read_dir}/Obs_N{self.N_truth}_freq{self.obs_freq}_nobs{self.nobs}_err{self.obs_err:.1e}.nc'
        
        return xr.open_dataset(obs_file)
    
    def init_DA(self, DA_start=0, ic_seed=0):
        """Initialize DA experiment"""
        self.ens = L96Ensemble([Lorenz96Model(N=self.N_DA, F=self.F, dt=self.dt) 
                               for _ in range(self.nens)])
        self.obs_ds = self.read_obs()
        
        if DA_start == 0:
            ic_file = f'{self.read_dir}/IC_x_N{self.N_DA}_ens{self.nens}.nc'
            if not exists(ic_file):
                self.ens_spinup()
            x_init = xr.open_dataarray(ic_file)
            
            for i, model in enumerate(self.ens.models):
                model.x = x_init[(i + ic_seed) % len(x_init)].values
        
        # Setup localization if needed
        if self.DA_method in ['EnKF', 'UnetKF'] and self.use_localization:
            self.W = self.compute_localization()
        
        return self.ens
    
    def compute_localization(self):
        """Compute Gaspari-Cohn localization matrix"""
        N = self.N_DA
        W = np.zeros((N, N))
        
        for i in range(N):
            for j in range(N):
                dist = min(abs(i - j), N - abs(i - j))  # Periodic distance
                W[i, j] = gaspari_cohn(dist, self.loc_radius)
        
        return W
    
    def assimilation(self, forecast, obs_cycle_idx, **kwargs):
        """
        Perform data assimilation
        
        Args:
            forecast: xarray with dimensions (model, time, dim) from run_for_steps
            obs_cycle_idx: which observation/DA cycle this is (0, 1, 2, ...)
        """
        # Extract prior as numpy array (last timestep)
        if isinstance(forecast, xr.DataArray):
            prior = forecast[-1, :, :].values  # Shape: (nens, N)
        else:
            prior = forecast[-1, :, :]  # Already numpy array
        
        # Get observations for this cycle
        H, R_obs = self.create_obs_operator(obs_cycle_idx)
        obs_x = self.obs_ds.x[obs_cycle_idx].values
        
        if self.DA_method == 'EnKF':
            # Ensemble Kalman Filter
            if self.inflate[0] > 1.0001:
                prior = ens_inflate(prior, prior, 1, self.inflate[0])
            
            B_ens = calculate_cov(prior)
            
            # Save full B matrix if requested
            if self.save_B:
                Path(f'{self.save_dir}/{self.file_name()}').mkdir(exist_ok=True)
                B_filename = f'{self.save_dir}/{self.file_name()}/B_ens_cycle{obs_cycle_idx:04d}.nc'
                B_ens_da = xr.DataArray(B_ens, dims=['dim', 'dim_d'])
                B_ens_ds = xr.Dataset({'B_ens': B_ens_da})
                B_ens_ds.to_netcdf(B_filename)
            
            # Apply localization only if use_localization is True
            if self.use_localization and hasattr(self, 'W'):
                B_ens_loc = B_ens * self.W
                # print(f"Cycle {obs_cycle_idx}: B_ens rank = {matrix_rank(B_ens)}, B_ens_loc rank = {matrix_rank(B_ens_loc)}")
                posterior = EnKF(prior, obs_x, H, R_obs, B_ens_loc)
            else:
                # print(f"Cycle {obs_cycle_idx}: B_ens rank = {matrix_rank(B_ens)} (no localization)")
                posterior = EnKF(prior, obs_x, H, R_obs, B_ens)
            
            if self.inflate[1] > 0.0001:
                posterior = ens_inflate(prior, posterior, 2, self.inflate[1])
                
        elif self.DA_method == 'UnetKF':
            # UNet-based Kalman Filter
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            ml_model = kwargs['ml_model'].to(device)
            ml_std = kwargs['std_file']
            
            # Prepare input for UNet (normalize)
            # For ensemble: use ensemble mean; for single member: use that member
            if self.nens > 1:
                prior_mean = prior.mean(axis=0)
            else:
                prior_mean = prior[0]
            
            x_input = prior_mean / ml_std.x_std.values
            
            # Predict B matrix from UNet
            x_tensor = torch.from_numpy(x_input[np.newaxis, np.newaxis, :]).float().to(device)
            B_pred = ml_model(x_tensor).to('cpu').detach().numpy()[0, :, :]
            B_pred = B_pred * ml_std.B_std.values
            
            # Save full B matrix if requested
            if self.save_B:
                Path(f'{self.save_dir}/{self.file_name()}').mkdir(exist_ok=True)
                B_filename = f'{self.save_dir}/{self.file_name()}/B_unet_cycle{obs_cycle_idx:04d}.nc'
                B_unet_da = xr.DataArray(B_pred, dims=['dim', 'dim_d'])
                B_unet_ds = xr.Dataset({'B_unet': B_unet_da})
                B_unet_ds.to_netcdf(B_filename)
            
            # Apply localization only if use_localization is True
            if self.use_localization and hasattr(self, 'W'):
                B_Unet = B_pred * self.W
                # print(f"Cycle {obs_cycle_idx}: B_Unet rank = {matrix_rank(B_pred)}, B_Unet_loc rank = {matrix_rank(B_Unet)}")
            else:
                B_Unet = B_pred
                # print(f"Cycle {obs_cycle_idx}: B_Unet rank = {matrix_rank(B_Unet)} (no localization)")
            
            B_Unet = B_Unet.astype(np.float64)

            if self.nens == 1:
                posterior = EnKF_mean(prior[0], obs_x, H, R_obs, B_Unet)
                posterior = posterior[np.newaxis, :]
            else:
                posterior = EnKF(prior, obs_x, H, R_obs, B_Unet)
            
            if self.inflate[1] > 0.0001:
                posterior = ens_inflate(prior, posterior, 2, self.inflate[1])
        else:
            posterior = prior
        
        return posterior
    
    def create_obs_operator(self, obs_cycle_idx):
        """
        Create observation operator H and error covariance R
        
        Args:
            obs_cycle_idx: DA cycle index (0, 1, 2, ...) matching obs dataset
        """
        nobs = self.nobs
        N = self.N_DA
        
        obs_idx = self.obs_ds.idx[obs_cycle_idx].values
        obs_err = self.obs_ds.err_std[obs_cycle_idx].values
        
        H = np.zeros((nobs, N))
        H[range(nobs), obs_idx] = 1.0
        
        R = np.diag(obs_err**2)
        
        return H, R
    
    def run_exp(self, DA_steps=1000, DA_start=0, ic_seed=0, output_str='', **kwargs):
        """
        Run data assimilation experiment
        
        Following the paper: saves both forecast (prior) and analysis (posterior)
        """
        self.init_DA(DA_start, ic_seed=ic_seed)       
        # Setup kwargs for UnetKF if needed
        DA_kwargs = {}
        if self.DA_method == 'UnetKF':
            DA_kwargs['ml_model'] = kwargs['ml_model']
            DA_kwargs['std_file'] = kwargs['ml_std_ds']
        
        n_cycles = DA_steps // self.DA_freq
        forecast_results = []  # Store forecasts (before DA)
        analysis_results = []  # Store analyses (after DA)
        
        for cycle in tqdm(range(n_cycles), desc=f"Running {self.DA_method}"):
            # Forecast phase: run model forward
            forecast = self.ens.run_for_steps(self.DA_freq, save_every=self.DA_freq)
            # Store forecast (last timestep before assimilation)
            # Extract as numpy array
            if isinstance(forecast, xr.DataArray):
                forecast_state = forecast[-1, :, :].values  # Shape: (nens, N)
            else:
                forecast_state = forecast[-1, :, :]
            
            forecast_results.append(forecast_state)
            
            if self.DA_method != 'NoDA':
                # Analysis phase: assimilate observations
                analysis = self.assimilation(forecast, cycle, **DA_kwargs)
                
                # Update ensemble with analysis
                for i, model in enumerate(self.ens.models):
                    model.x = analysis[i]
                
                # Store analysis
                analysis_results.append(analysis)
            else:
                # No DA: analysis = forecast
                analysis_results.append(forecast_state)
        
        # Convert lists to arrays
        forecast_array = np.array(forecast_results)  # Shape: (n_cycles, nens, N)
        analysis_array = np.array(analysis_results)  # Shape: (n_cycles, nens, N)
        
        # Create xarray datasets
        times = np.arange(n_cycles) * self.DA_freq * self.dt
        
        # Transpose to match expected dimensions: (time, model, dim)
        forecast_xr = xr.DataArray(
            forecast_array,
            dims=['time', 'model', 'dim'],
            coords={'time': times, 'model': range(self.nens), 'dim': range(self.N_DA)}
        )
        
        analysis_xr = xr.DataArray(
            analysis_array,
            dims=['time', 'model', 'dim'],
            coords={'time': times, 'model': range(self.nens), 'dim': range(self.N_DA)}
        )
        
        # Compute ensemble means
        forecast_mean = forecast_xr.mean(dim='model')
        analysis_mean = analysis_xr.mean(dim='model')
        
        # Save results
        target_dir = f'{self.save_dir}{output_str}'
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        
        file_name = self.file_name()
        
        # Save ensemble mean (analysis) - this is the main output
        mean_file = f'{target_dir}/EnsMean_{file_name}.nc'
        ds_mean = xr.Dataset({'x': analysis_mean})
        ds_mean.to_netcdf(mean_file, mode='w')
        
        # Save ensemble spread if ensemble size > 1
        if self.nens > 1:
            analysis_std = analysis_xr.std(dim='model')
            std_file = f'{target_dir}/EnsStd_{file_name}.nc'
            ds_std = xr.Dataset({'x': analysis_std})
            ds_std.to_netcdf(std_file, mode='w')
            
            # Optionally save forecast mean and std for diagnostics
            forecast_mean_file = f'{target_dir}/ForecastMean_{file_name}.nc'
            ds_forecast_mean = xr.Dataset({'x': forecast_mean})
            ds_forecast_mean.to_netcdf(forecast_mean_file, mode='w')
    
    def file_name(self):
        """Generate experiment filename"""
        if self.DA_method == 'NoDA':
            return f'Control_N{self.N_DA}'
        elif self.DA_method == 'EnKF':
            return f'EnKF_N{self.N_DA}_ens{self.nens}_freq{self.DA_freq}_relax{self.inflate[1]:.2f}_loc{self.loc_radius:.1f}_nobs{self.nobs}_err{self.obs_err:.1e}'
        elif self.DA_method == 'UnetKF':
            return f'UnetKF_N{self.N_DA}_ens{self.nens}_freq{self.DA_freq}_relax{self.inflate[1]:.2f}_loc{self.loc_radius:.1f}_nobs{self.nobs}_err{self.obs_err:.1e}'
        else:
            return f'{self.DA_method}_N{self.N_DA}'

    def plot_truth(self, steps, cut=0, save=False):
        """
        Creates a high-quality contour plot of the Lorenz-96 truth trajectory.
        """
        truth_file_path = f'{self.save_dir}/Truth_N{self.N_truth}_{steps}steps.nc'

        if not os.path.exists(truth_file_path):
            print(f"Truth file not found: {truth_file_path}")
            return

        with xr.open_dataset(truth_file_path) as ds:
            # Apply initial cut to remove spin-up if needed
            x_vals = ds.x.values[cut:, :]
            t_vals = ds.time.values[cut:]
            x_indices = np.arange(x_vals.shape[1])

        # Create the figure
        plt.figure(figsize=(12, 8))
        
        # Use contourf to get the smooth "flowing" look from your reference
        # 20-30 levels usually provides a nice balance of detail and smoothness
        contour = plt.contourf(x_indices, t_vals, x_vals, levels=25, cmap='RdBu_r')
        
        # Add a subtle line contour on top to match your black-and-white reference
        plt.contour(x_indices, t_vals, x_vals, levels=25, colors='black', linewidths=0.2, alpha=0.5)

        # --- THE FIX FOR SCALING ---
        # Forces the plot to be readable regardless of step count
        plt.gca().set_aspect('auto') 
        
        # To make it look like the reference (waves moving 'forward' in time)
        # usually time on Y increases upwards, but you can invert if preferred
        plt.ylabel('Model Time')
        plt.xlabel('Spatial Index (n)')
        plt.title(f'Lorenz-96 Hovmöller Diagram (F={self.F})')
        
        plt.colorbar(contour, label='State value (u)')
        plt.tight_layout()

        if save:
            plt.savefig(f"{self.save_dir}/truth_contour_plot.png", dpi=300)
        
        plt.show()

class L96Ensemble:
    """Ensemble of Lorenz 96 models"""
    
    def __init__(self, models):
        for m in models:
            m.x = np.array(m.x, copy=True)
        self.models = models
        self.ens = len(models)
    
    def step_forward(self):
        """Step all models forward"""
        for m in self.models:
            m.step_forward()
    
    def run_for_steps(self, steps, save_every=1):
        """Run ensemble for specified steps"""
        results = []
        
        for i in range(steps):
            self.step_forward()
            if (i + 1) % save_every == 0:
                x_all = np.array([m.x for m in self.models])
                results.append(x_all)
        
        x_array = np.array(results)  # Shape: (time, model, dim)
        times = np.arange(len(results)) * save_every * self.models[0].dt
        
        return xr.DataArray(x_array, 
                           dims=['time', 'model', 'dim'],
                           coords={'time': times,
                                  'model': range(self.ens),
                                  'dim': range(self.models[0].N)})


class L96_UNet(torch.nn.Module):
            """
            U-Net for predicting full B matrix from L96 state
            Simplified 1D version of the paper's 2D U-Net
            """
            def __init__(self, channels_in=1, channels_out=2, width=16, N = 40):
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


# Data assimilation functions
@jit(nopython=True)
def calculate_cov(data):
    """Calculate covariance matrix"""
    return np.cov(data.T)

@jit(nopython=True)
def EnKF(prior, obs, H, R, B):
    """Ensemble Kalman Filter update"""
    nens, N = prior.shape
    nobs = obs.shape[0]
    
    # Kalman gain
    D = H @ B @ H.T + R
    K = B @ H.T @ np.linalg.inv(D)
    
    # Perturb observations
    obs_ens = obs.repeat(nens).reshape(nobs, nens) + \
              np.sqrt(R) @ np.random.standard_normal((nobs, nens))
    
    # Analysis
    posterior = prior.T + K @ (obs_ens - H @ prior.T)
    
    return posterior.T

@jit(nopython=True)
def EnKF_mean(prior, obs, H, R, B):
    """EnKF for single member (deterministic)"""
    nobs = obs.shape[0]
    
    # Kalman gain
    D = H @ B @ H.T + R
    K = B @ H.T @ np.linalg.inv(D)
    
    # Analysis (no perturbation for single member)
    posterior = prior + K @ (obs - H @ prior)
    
    return posterior

@jit(nopython=True)
def ens_inflate(prior, posterior, opt, factor):
    """Ensemble inflation - Numba Compatible"""
    nens, N = prior.shape
    inflated = np.zeros(prior.shape)
    
    # Calculate means without relying on .mean(axis=0) if it fails
    # We can use np.sum / nens which is very stable in Numba
    if opt == 1:  # Prior inflation
        # Manual mean calculation for max Numba compatibility
        mean_prior = np.zeros(N)
        for j in range(N):
            mean_prior[j] = np.sum(prior[:, j]) / nens
            
        for i in range(nens):
            inflated[i, :] = mean_prior + factor * (prior[i, :] - mean_prior)
            
    elif opt == 2:  # Relaxation to prior spread (RTPS)
        mean_prior = np.zeros(N)
        mean_post = np.zeros(N)
        for j in range(N):
            mean_prior[j] = np.sum(prior[:, j]) / nens
            mean_post[j] = np.sum(posterior[:, j]) / nens
            
        for i in range(nens):
            inflated[i, :] = mean_post + (1 - factor) * (posterior[i, :] - mean_post) + \
                             factor * (prior[i, :] - mean_prior)
    
    return inflated
# def ens_inflate(prior, posterior, opt, factor):
#     """Ensemble inflation"""
#     nens, N = prior.shape
#     inflated = np.zeros(prior.shape)
    
#     if opt == 1:  # Prior inflation
#         mean_prior = prior.mean(axis=0).repeat(nens).reshape(N, nens).T
#         inflated = prior + factor * (prior - mean_prior)
#     elif opt == 2:  # Relaxation to prior spread
#         mean_prior = prior.mean(axis=0).repeat(nens).reshape(N, nens).T
#         mean_post = posterior.mean(axis=0).repeat(nens).reshape(N, nens).T
#         inflated = mean_post + (1 - factor) * (posterior - mean_post) + \
#                    factor * (prior - mean_prior)
    
#     return inflated

def gaspari_cohn(distance, radius):
    """Gaspari-Cohn localization function"""
    if distance == 0:
        return 1.0
    if radius == 0:
        return 0.0
    
    ratio = distance / radius
    
    if ratio <= 1:
        return -ratio**5/4 + ratio**4/2 + 5*ratio**3/8 - 5*ratio**2/3 + 1
    elif ratio <= 2:
        return ratio**5/12 - ratio**4/2 + 5*ratio**3/8 + 5*ratio**2/3 - 5*ratio + 4 - 2/(3*ratio)
    else:
        return 0.0