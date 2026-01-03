import os
import torch
import numpy as np
import xarray as xr
import DA_core as DA
import ML_core as ML
from torch import optim
from matplotlib import pyplot as plt
import logging

###cut down on what is printed
# Silence all loggers
logging.getLogger().setLevel(logging.WARNING)
# Specifically target the pyqg or DA loggers if the above isn't enough
logging.getLogger('pyqg').setLevel(logging.WARNING)
logging.getLogger('DA').setLevel(logging.WARNING)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
two_yrs = 365*2
nine_yrs = 365*9
DAYS_TEST = 365     # 1 year for the final UnetKF run
BASE_DIR = './data'

#-----------------------#
#GENERATE TRAINING DATA
#-----------------------#
####Experiments to Run####
DA_exp_L20 = {'nens': 20,
    'DA_method': 'EnKF',
    'Nx_DA': 32,
    'Nx_truth': 32,
    'obs_freq': 10,
    'obs_err': [1, -5, 5, -7],
    'nobs': [40, 40],
    'R_W': 100,  
    'B_loc': None,
    'DA_freq': 10,
    'save_B': True,  # Required to train the UNet
    'inflate': [1, 0.6],
    'delta_days': 50}

DA_exp_L80 = {
    'nens': 80,
    'DA_method': 'EnKF',
    'Nx_DA': 32,
    'Nx_truth': 32,
    'obs_freq': 10,
    'obs_err': [1, -5, 5, -7],
    'nobs': [40, 40],
    'R_W': 100,  
    'B_loc': None,
    'DA_freq': 10,
    'save_B': True,
    'inflate': [1, 0.5], # Relaxation factor 0.5
    'delta_days': 50}

DA_exp_L1280 = {
    'nens': 1280,
    'DA_method': 'EnKF',
    'Nx_DA': 32,
    'Nx_truth': 32,
    'obs_freq': 10,
    'obs_err': [1, -5, 5, -7],
    'nobs': [40, 40],
    'R_W': 100,  
    'B_loc': None,
    'DA_freq': 10,
    'save_B': True,
    'inflate': [1, 0.45], # Relaxation factor 0.45
    'delta_days': 50}

DA_exp_H20 = {
    'nens': 20,
    'DA_method': 'EnKF',
    'Nx_DA': 128, # High-resolution grid
    'Nx_truth': 128,
    'obs_freq': 10,
    'obs_err': [1, -5, 5, -7],
    'nobs': [40, 40],
    'R_W': 100,  
    'B_loc': None,
    'DA_freq': 10,
    'save_B': True,
    'inflate': [1, 0.6], # Relaxation factor 0.6
    'delta_days': 50}

DA_exp_H80 = {
    'nens': 80,
    'DA_method': 'EnKF',
    'Nx_DA': 128, # High-resolution grid
    'Nx_truth': 128,
    'obs_freq': 10,
    'obs_err': [1, -5, 5, -7],
    'nobs': [40, 40],
    'R_W': 100,  
    'B_loc': None,
    'DA_freq': 10,
    'save_B': True,
    'inflate': [1, 0.45], # Relaxation factor 0.45
    'delta_days': 50}

experiments = [DA_exp_L20]
DAYS_TRAIN = [two_yrs]
exp_name = ['L20']
# DAYS_TRAIN = [two_yrs, nine_yrs, nine_yrs, nine_yrs, two_yrs, nine_yrs]  # training for all tests

for i in range(len(experiments)):
	print('------------------')
	print(f'Running Experiment {exp_name[i]}')
	print('------------------')

	mean_folder = (f'EnsMean_EnKF_Nx{experiments[i]["Nx_DA"]}_from_Nx{experiments[i]["Nx_DA"]}_ens{experiments[i]["nens"]}_freq10_relax{experiments[i]["inflate"][1]}_R100_nobs40_40_err1E-5_5E-7')
	mean_file_path = f'./data/{exp_name[i]}/{mean_folder}.nc'

	if not os.path.exists(mean_file_path):
		print()
		print('>>>>Phase 1: Generate Training Data via EnKF')
		DA_train = DA.DA_exp(exp_id = exp_name[i], **experiments[i])

		#add in logic for skipping this if data already there
		DA_train.generate_truth(years=5)
		DA_train.generate_obs(years=5)

		DA_train.ens_spinup(years=3)
		DA_train.run_exp(DA_days=DAYS_TRAIN[i], DA_start=0,ic_ens=experiments[i]['nens'])
	else:
		print()
		print('>>>>Phase 1: SKIPPING (Data already exists)')

	print()
	print('>>>>Phase 2: Preprocess to Train UNet')
	b_folder = (f'EnKF_Nx{experiments[i]["Nx_DA"]}_from_Nx{experiments[i]["Nx_DA"]}_ens{experiments[i]["nens"]}_freq10_relax{experiments[i]["inflate"][1]}_R100_nobs40_40_err1E-5_5E-7')
	mean_folder = (f'EnsMean_EnKF_Nx{experiments[i]["Nx_DA"]}_from_Nx{experiments[i]["Nx_DA"]}_ens{experiments[i]["nens"]}_freq10_relax{experiments[i]["inflate"][1]}_R100_nobs40_40_err1E-5_5E-7')
	B_ens_ds = xr.open_mfdataset(f'./data/{exp_name[i]}/{b_folder}/B_ens_day*.nc',combine='nested',concat_dim='time')
	mean_ds = xr.open_dataset(f'./data/{exp_name[i]}/{mean_folder}.nc').load()

	# spin_up_index = int(len(mean_ds.time) * 0.2) 
	# end_index = len(mean_ds.time)
	# step = experiments[i]['DA_freq'] 

	# DA_days = slice(spin_up_index, end_index, step)

	# print(f"Shape of B_ens: {B_ens_ds.B_ens.shape}")
	# print(f"Sample of raw B_ens values (channel 0,0): {B_ens_ds.B_ens.isel(lev=0, lev_d=0, time=0, y=16, x=16, y_d=slice(0,5), x_d=slice(0,5)).values}")
	# print(f"Sample of raw B_ens values (channel 0,1): {B_ens_ds.B_ens.isel(lev=0, lev_d=1, time=0, y=16, x=16, y_d=slice(0,5), x_d=slice(0,5)).values}")
	# print(f"Mean |B_ens| (0,0): {np.mean(np.abs(B_ens_ds.B_ens.isel(lev=0, lev_d=0).values)):.2e}")
	# print(f"Mean |B_ens| (0,1): {np.mean(np.abs(B_ens_ds.B_ens.isel(lev=0, lev_d=1).values)):.2e}")
	# print(f"Mean |B_ens| (1,1): {np.mean(np.abs(B_ens_ds.B_ens.isel(lev=1, lev_d=1).values)):.2e}")

	# print(f"Raw B_ens_ds (0,1) stats:")
	# print(f"  Mean: {np.mean(np.abs(B_ens_ds.B_ens.isel(lev=0, lev_d=1).values)):.2e}")
	# print(f"  Std: {np.std(B_ens_ds.B_ens.isel(lev=0, lev_d=1).values):.2e}")
	# print(f"  Max: {np.max(np.abs(B_ens_ds.B_ens.isel(lev=0, lev_d=1).values)):.2e}")
	# print(f"  99th percentile: {np.percentile(np.abs(B_ens_ds.B_ens.isel(lev=0, lev_d=1).values), 99):.2e}")
	# print(f"  Values > 1e-11: {np.sum(np.abs(B_ens_ds.B_ens.isel(lev=0, lev_d=1).values) > 1e-11)}")
	###might need to change later to update with each experiment
	B_size=16
	B_start=0

	n_samples = len(B_ens_ds.time)
	spin_up = int(n_samples * 0.3) 
	# print('n_samples:', n_samples)
	# print('spin up:', spin_up)
	step = experiments[i]['DA_freq']
	DA_days_B = slice(spin_up, n_samples, 1)
	DA_days_q = slice(spin_up * step, n_samples * step, step)

	# print(f"Number of q samples: {len(range(*DA_days_q.indices(n_samples*step)))}")
	# print(f"Number of B samples: {len(range(*DA_days_B.indices(n_samples)))}")

	i_x=slice(0,experiments[i]['Nx_DA'])
	i_y=slice(0,experiments[i]['Nx_DA'])
	# train_y = slice(B_start, B_start + B_size)
	# train_x = slice(B_start, B_start + B_size)
	# DA_it = slice(
	# 	int((DA_days_B.start - step + 1) / step), 
	# 	int((DA_days_B.stop - step + 1) / step) + 1
	# 	)

	B_ens = B_ens_ds.B_ens.isel(time=DA_days_B, y=i_y, x=i_x)
	q_full=mean_ds.q.isel(time=DA_days_q,y=i_y,x=i_x)

	B_std=np.empty((2,2))
	B_std[0,0]=B_ens_ds.B_ens.isel(lev=0,lev_d=0).std()
	B_std[0,1]=B_ens_ds.B_ens.isel(lev=0,lev_d=1).std()
	B_std[1,0]=B_std[0,1]
	B_std[1,1]=B_ens_ds.B_ens.isel(lev=1,lev_d=1).std()

	# print(B_ens_ds.B_ens.isel(lev=0, lev_d=1).max().values)
	# print(f"Mean of raw B_ens Channel 1: {np.mean(np.abs(B_ens_ds.B_ens.isel(lev=0, lev_d=1).values)):.2e}")
	# print(f"B_std Ch1: {B_std[0,1]:.2e}")

	q_std=np.zeros((2,1))
	q_std[0]=q_full.isel(lev=0).std()
	q_std[1]=q_full.isel(lev=1).std()

	# print(f"Calculated B_std: {B_std.flatten()}")
	# print(f"Calculated q_std: {q_std.flatten()}")

	ml_std_ds=xr.Dataset({'B_std':xr.DataArray(B_std,coords=[mean_ds.lev,mean_ds.lev]),
		'q_std':xr.DataArray(q_std.squeeze(),coords=[mean_ds.lev])})
	ml_std_ds.to_netcdf(f'./data/{exp_name[i]}/std.nc')

	B_stacked=B_ens.compute().stack(sample=('time','y','x')).transpose('sample',...)

	B_data=np.empty((len(B_ens.time)*len(B_ens.y)*len(B_ens.x),3,len(B_ens.y_d),len(B_ens.x_d)))
	# B_data=np.empty((len(B_ens.time)*B_size*B_size,3,len(B_ens.y_d),len(B_ens.x_d)))
	B_data[:,0,...]=B_stacked[:,0,0,...]/ml_std_ds.B_std[0,0].data
	B_data[:,1,...]=B_stacked[:,0,1,...]/ml_std_ds.B_std[0,1].data
	B_data[:,2,...]=B_stacked[:,1,1,...]/ml_std_ds.B_std[1,1].data

	# print(f"B_stacked shape: {B_stacked.shape}")
	# print(f"B_data shape: {B_data.shape}")
	# print(f"B_stacked[:,0,0,...] mean: {np.mean(np.abs(B_stacked[:,0,0,...].values)):.2e}")
	# print(f"B_stacked[:,0,1,...] mean: {np.mean(np.abs(B_stacked[:,0,1,...].values)):.2e}")
	# print(f"B_stacked[:,1,1,...] mean: {np.mean(np.abs(B_stacked[:,1,1,...].values)):.2e}")
	# print(f"After division:")
	# print(f"Ch 0 mean: {np.mean(np.abs(B_data[:,0,...])):.2f}, max: {np.max(np.abs(B_data[:,0,...])):.2f}")
	# print(f"Ch 1 mean: {np.mean(np.abs(B_data[:,1,...])):.2f}, max: {np.max(np.abs(B_data[:,1,...])):.2f}")
	# print(f"Ch 2 mean: {np.mean(np.abs(B_data[:,2,...])):.2f}, max: {np.max(np.abs(B_data[:,2,...])):.2f}")

	# raw_val = np.abs(B_stacked[:,0,0,...].values).mean()
	# norm_val = np.abs(B_data[:,0,...]).mean()
	# print(f"Channel 0: Raw Mean ({raw_val:.2e}) / Std ({ml_std_ds.B_std[0,0].values:.2e}) = {norm_val:.2f}")

	q_subsampled = mean_ds.q.isel(time=DA_days_q, y=i_y, x=i_x).compute()
	q_local=np.empty((len(q_subsampled.time),len(q_subsampled.lev),len(q_subsampled.y),len(q_subsampled.x),len(B_ens.y_d),len(B_ens.x_d)))
	for j in range(len(q_subsampled.x)):
		for k in range(len(q_subsampled.y)):
			q_local[:,:,k,j,:,:]=DA.localize_q(q_subsampled,k,j,experiments[i]['Nx_DA'],int(len(B_ens.x_d)/2))

	q_local=q_local.transpose([0,2,3,1,4,5])
	q_data=q_local.reshape((len(q_subsampled.time)*len(q_subsampled.y)*len(q_subsampled.x),len(q_subsampled.lev),len(B_ens.y_d),len(B_ens.x_d)))
	q_data[:,0,...]=q_data[:,0,...]/ml_std_ds.q_std[0].data
	q_data[:,1,...]=q_data[:,1,...]/ml_std_ds.q_std[1].data

	# Identify if any specific index is massive
	# max_val = np.max(np.abs(B_data))
	# min_val = np.min(np.abs(B_data))
	# print(f"Max in B_data: {max_val:.2f}, Min: {min_val:.2f}")

	# Check if the standard deviation itself is nearly zero
	# print(f"Smallest B_std used: {min(std00, std01, std11):.2e}")
	# print(f"Ch 0 Max: {np.max(np.abs(B_data[:,0,...])):.2f}")
	# print(f"Ch 1 Max: {np.max(np.abs(B_data[:,1,...])):.2f}")
	# print(f"Ch 2 Max: {np.max(np.abs(B_data[:,2,...])):.2f}")

	# print(f"B_data mean: {np.mean(np.abs(B_data))}")
	# print(f"q_data mean: {np.mean(np.abs(q_data))}")

	q_unet=q_data[...,B_start:B_start+B_size,B_start:B_start+B_size]
	B_unet=B_data[...,B_start:B_start+B_size,B_start:B_start+B_size]
	B_shape=B_unet.shape
	q_shape=q_unet.shape

	print()
	print('>>>>Phase 3: Training UNet')

	n_total=B_shape[0]
	n_train=int(n_total*0.8)
	in_ch=[0,1]
	out_ch=[0,1,2]

	train_ds=ML.Dataset(q_unet[0:n_train,...],B_unet[0:n_train,...],DEVICE)
	valid_ds=ML.Dataset(q_unet[n_train:,...],B_unet[n_train:,...],DEVICE)

	params = {'batch_size':16,'num_workers':0,'shuffle':True}
	training_generator = torch.utils.data.DataLoader(train_ds, **params)
	validation_generator = torch.utils.data.DataLoader(valid_ds, **params)

	model=ML.Unet_2L(in_ch=len(in_ch),out_ch=len(out_ch))
	model=model.to(DEVICE)

	criterion = torch.nn.RMSELoss() # RMSE loss function
	optimizer = optim.Adam(model.parameters(), lr=0.002)

	model=model.double()

	n_epochs = 200 #Number of epocs
	validation_loss = list()
	train_loss = list()
	start_epoch=0
	if start_epoch>0:
		model_file='./data/{}/unet_epoch{}_in{}_out{}_B{}.pt'.format(exp_name[i],start_epoch,''.join(map(str,in_ch)),''.join(map(str,out_ch)),B_size)
		model.load_state_dict(torch.load(model_file,map_location=torch.device('cpu')))

	for epoch in range(start_epoch+1, n_epochs + 1):
		train_loss.append(ML.train_model(model,criterion,training_generator,optimizer,DEVICE))
		validation_loss.append(ML.test_model(model,criterion,validation_generator,optimizer,DEVICE))
		torch.save(model.state_dict(), './data/{}/unet_epoch{}_in{}_out{}_B{}.pt'.\
        	format(exp_name[i],epoch,''.join(map(str,in_ch)),''.join(map(str,out_ch)),B_size))



