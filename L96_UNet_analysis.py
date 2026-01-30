import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import glob
import re
import xarray as xr
import torch
import torch.nn as nn
import L96_DA_core as L96

##Experiments
# Experiment with 20 ensemble members
DA_exp_L20 = {
    'N_truth': 40,
    'N_DA': 40,
    'nens': 20,
    'DA_method': 'EnKF',
    'obs_freq': 2,          # Observe every 4 time steps
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
    'obs_freq': 2,          # Observe every 4 time steps
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
    'obs_freq': 2,          # Observe every 4 time steps
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
    'obs_freq': 2,          # Observe every 4 time steps
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
    'obs_freq': 2,          # Observe every 4 time steps
    'obs_err': 1.0,         # Observation error std
    'nobs': 20,             # Number of observations per cycle
    'loc_radius': 5.0,      # Localization radius
    'DA_freq': 4,           # DA cycle frequency
    'save_B': True,         # Save B matrices for training
    'use_localization': False,  # Use FULL B matrix (no localization)
    'inflate': [1.05, 0.0],  # [prior_inflation, relaxation]
    'F': 8.0,               # Lorenz 96 forcing
    'dt': 0.01}             # Time step

# Load CSV
base_dir = './data/lorenz96'
experiments = [DA_exp_L20, DA_exp_L20_w16,DA_exp_L20_w4,DA_exp_L20_EAKF,DA_exp_L20_ETKF]
exp_name = ['L20', 'L20_w16','L20_w4','L20_EAKF','L20_ETKF']
widths = [40,16,4,4,4]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for i in range(len(exp_name)):

	exp = exp_name[i]
	df = pd.read_csv(f"{base_dir}/{exp}/losses.csv")

	###training vs validation loss for all epochs
	plt.figure(figsize=(7, 5))

	plt.plot(df.index, df["train_loss"], label="Training Loss")
	plt.plot(df.index, df["valid_loss"], label="Validation Loss")

	plt.xlabel("Epoch")
	plt.ylabel("MSE Loss")
	plt.title("Training vs Validation Loss")
	plt.legend()
	plt.grid(True)

	plt.tight_layout()
	plt.savefig(f'./{base_dir}/{exp_name[i]}/losses_plt.png')
	plt.show()

	###covariance of Kalman Filter
	# Construct expected filenames
	print('importing mean file')
	mean_folder = (f'EnsMean_{experiments[i]["DA_method"]}_N{experiments[i]["N_DA"]}_ens{experiments[i]["nens"]}_'
		f'freq{experiments[i]["DA_freq"]}_relax{experiments[i]["inflate"][1]:.2f}_'
		f'loc{experiments[i]["loc_radius"]:.1f}_nobs{experiments[i]["nobs"]}_'
		f'err{experiments[i]["obs_err"]:.1e}')

	print('importing covariance files')
	b_folder = (f'{experiments[i]["DA_method"]}_N{experiments[i]["N_DA"]}_ens{experiments[i]["nens"]}_'
		f'freq{experiments[i]["DA_freq"]}_relax{experiments[i]["inflate"][1]:.2f}_'
		f'loc{experiments[i]["loc_radius"]:.1f}_nobs{experiments[i]["nobs"]}_'
		f'err{experiments[i]["obs_err"]:.1e}')

	# Load B matrices and ensemble mean
	B_ens_ds = xr.open_mfdataset(
		f'{base_dir}/{exp_name[i]}/{b_folder}/B_ens_cycle*.nc',
		combine='nested',
		concat_dim='cycle'
		)
	mean_ds = xr.open_dataset(
		f'{base_dir}/{exp_name[i]}/{mean_folder}.nc'
		).load()

	# Split data: use second half for training (after spin-up)
	n_samples = len(B_ens_ds.cycle)
	spin_up = int(n_samples * 0.5)

	# Extract B matrices and state data
	B_ens = B_ens_ds.B_ens.isel(cycle=slice(spin_up, None))
	num_cycles = B_ens.shape[0]
	fig, ax = plt.subplots(figsize=(8, 6))
	vlimit = np.percentile(np.abs(B_ens), 95)
	im = ax.imshow(B_ens[0], cmap='bwr', animated=True,
		vmin=-vlimit, vmax=vlimit, interpolation='nearest')

	ax.set_title(f"B_ens Matrix Evolution (Post Spin-up)")
	plt.colorbar(im, label='Covariance Value')

	def update(frame):
		im.set_array(B_ens[frame])
		ax.set_title(f"B_ens Matrix - Cycle {frame}") # Adjust title to show actual cycle index

		return [im]

	print('making animation')
	ani = FuncAnimation(fig, update, frames=num_cycles,interval=150, blit=True)
	ani.save(f"./{base_dir}/{exp_name[i]}/B_ens_Matrix_ w{widths[i]}.gif", writer='pillow', fps=10)

	plt.show()

	####Covariance of EnKF and UNet Predicted
	x_data = mean_ds.x.isel(time=slice(spin_up, None))
	B_std = B_ens.std().values
	x_std = x_data.std().values
	B_data = (B_ens.values / B_std).astype(np.float32)
	x_norm = (x_data.values / x_std).astype(np.float32)
	n_train_samples = B_data.shape[0]
	N = experiments[i]['N_DA']

	x_unet = x_norm[:, np.newaxis, :]
	B_unet = B_data.astype(np.float32)

	n_total = x_unet.shape[0]
	n_train_split = int(n_total * 0.8)

	model = model = L96.L96_UNet(N = N, width = widths[i])
	model = model.float()

	# Select one sample to track
	sample_idx = 0
	# We add a dummy batch dimension for the model: [1, H, W]
	x_valid = torch.from_numpy(x_unet[n_train_split:]).float()
	B_valid = torch.from_numpy(B_unet[n_train_split:]).float()

	# 2. Sort Checkpoints
	checkpoint_files = glob.glob(f'{base_dir}/{exp_name[i]}/unet_epoch*.pt')
	checkpoint_files.sort(key=lambda f: int(re.findall(r'\d+', f)[-1]))

	history_preds = []
	epochs = []

	model.to(DEVICE)
	model.eval()

	with torch.no_grad():
		for f in checkpoint_files:
			epochs.append(re.findall(r'\d+', f)[-1])
			model.load_state_dict(torch.load(f, map_location=DEVICE))

			pred = model(x_valid) 
			# squeeze() removes all dimensions of size 1, leaving just [H, W]
			history_preds.append(pred.squeeze().cpu().numpy())

	# 3. Create Three-Panel Animation (Truth, Prediction, Error)
	fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

	# Common scale for Truth and Pred
	vlimit = np.max(np.abs(B_valid.cpu().numpy()))

	truth_to_plot = B_valid[0].cpu().numpy() 
	im1 = ax1.imshow(truth_to_plot, cmap='bwr', vmin=-vlimit, vmax=vlimit)
	ax1.set_title("Ground Truth")

	first_epoch_first_sample = history_preds[0][0]
	im2 = ax2.imshow(first_epoch_first_sample, cmap='bwr', vmin=-vlimit, vmax=vlimit)
	ax2.set_title(f"UNet Pred (Epoch {epochs[0]})")

	# Residual plot (Difference) - uses a different scale to highlight errors
	residual = truth_to_plot- first_epoch_first_sample
	im3 = ax3.imshow(residual, cmap='PuOr', vmin=-vlimit/2, vmax=vlimit/2)
	ax3.set_title("Residual (Truth - Pred)")

	plt.colorbar(im1, ax=ax1)
	plt.colorbar(im2, ax=ax2)
	plt.colorbar(im3, ax=ax3)

	def update(frame):
		current_truth = B_valid[frame].cpu().numpy()
		current_pred = history_preds[frame][0] # Taking first sample of the batch

		im1.set_array(current_truth)
		im2.set_array(current_pred)
		im3.set_array(current_truth - current_pred) # Update residual based on current truth

		ax1.set_title(f"Truth (Cycle {frame})")
		ax2.set_title(f"UNet Pred (Epoch {epochs[frame]})")

		return [im1, im2, im3]

	ani = FuncAnimation(fig, update, frames=len(history_preds), interval=250, blit=True)
	plt.tight_layout()
	plt.show()