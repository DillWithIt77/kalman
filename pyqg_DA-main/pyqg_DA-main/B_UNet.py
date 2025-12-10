import xarray as xr
import numpy as np
import importlib
from matplotlib import pyplot as plt
import DA_core as DA
from glob import glob
import torch.utils.data as Data
from torch import optim
import torch
from torchsummary import summary
import ML_core as ML
from numpy.random import default_rng
import os
from os.path import exists
from torch.profiler import profile, record_function, ProfilerActivity

rng = default_rng()
# DA.read_data_dir='/scratch/cimes/feiyul/PyQG/data/training'
# DA.save_data_dir='/scratch/cimes/feiyul/PyQG/data/training'
# data_dir='/scratch/cimes/feiyul/PyQG/data'
# data_dir='/net2/fnl/PyQG/data'
DA.read_data_dir = './data/training'
DA.save_data_dir = './data/training'
data_dir = './data'


B_ens_kws={'cmap':'bwr','levels':np.linspace(-2.5E-11,2.5E-11,26),'extend':'both'}
B_ens_kws1={'cmap':'bwr','levels':np.linspace(-0.25E-11,0.25E-11,26),'extend':'both'}
q_kws={'cmap':'bwr','levels':np.linspace(-3.4E-5,3.4E-5,18),'extend':'both'}

importlib.reload(ML)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(device)

DA_paras={'nens':20,
          'DA_method':'EnKF',
          'Nx_DA':32,
          'Nx_truth':128,
          'obs_freq':10,
          'obs_err':[1,-5,5,-7],
          'DA_freq':10,
          'save_B':False,
          'nobs':[10,10],
          'R_W':150,
          'inflate':[1,0.5],
          'delta_days': 50}
DA_exp=DA.DA_exp(**DA_paras)
print(DA_exp.file_name())
# obs_ds=DA_exp.read_obs()
in_ch=[0,1]
out_ch=[0,1,2]

DA_paras1={'nens':320,
          'DA_method':'EnKF',
          'Nx_DA':64,
          'Nx_truth':128,
          'obs_freq':10,
          'obs_err':[1,-6,2,-8],
          'DA_freq':10,
          'save_B':False,
          'nobs':[100,0],
          'R_W':100,
          'inflate':[1,0.55],
          'delta_days': 50}
DA_exp1=DA.DA_exp(**DA_paras1)

# --- INSERT THIS BLOCK TO GENERATE THE NECESSARY OBS FILE ---
if not exists(DA_exp1.obs_name()):
    print(f"WARNING: Obs file for DA_exp1 ({DA_exp1.file_name()}) not found. Generating now (10-year truth run)...")
    # 1. Run the truth simulation for the necessary period (e.g., 10 years)
    DA_exp1.generate_truth(years=10) 
    # 2. Generate the observations that match DA_exp1's parameters
    DA_exp1.generate_obs(years=10)
    print("Observation generation complete for DA_exp1.")
else:
    print(f"INFO: Obs file for DA_exp1 ({DA_exp1.file_name()}) already exists.")
# -------------------------------------------------------------

print("Starting long EnKF run to generate U-Net training data (approx. 20 years)...")
# Run the 20-year (7300 days) EnKF simulation
DA_exp1.run_exp(DA_days=7300, DA_start=0)
print("EnKF training data generation complete.")

### Make direcotry for storing networks
os.makedirs('./ML/{}'.format(DA_exp.file_name()),exist_ok=True)

### Find time indices for the DA steps to select proper q and B data
DA_days=slice(729,1460,DA_exp.DA_freq)
DA_it=slice(int((DA_days.start-DA_exp.DA_freq+1)/DA_exp.DA_freq),int((DA_days.stop-DA_exp.DA_freq+1)/DA_exp.DA_freq)+1)
print(DA_days,DA_it)

### range of indices to select from q and B data
i_x=slice(0,DA_exp.Nx_DA)
i_y=slice(0,DA_exp.Nx_DA)

### size and starting index for the B data in U-Nets
B_size=20
B_start=0

### Read the saved covariance matrices from previous EnKF experiments
B_ens_ds=xr.open_dataset('{}/training/{}/B_ens.nc'.format(data_dir,DA_exp.file_name()))
B_ens=B_ens_ds.B_ens.isel(time=DA_it,y=i_y,x=i_x)
print(B_ens.shape)

### Read the saved ensemble-mean analysis q from previous EnKF experiments
mean_ds=DA_exp.read_mean().load()
q_full=mean_ds.q.isel(time=DA_days,y=i_y,x=i_x)
print(q_full.shape)

### Read or calculate standard deviations for normalization
if os.path.exists('./ML/{0}/std_{0}.nc'.format(DA_exp.file_name())):
    ml_std_ds=xr.open_dataset('./ML/{0}/std_{0}.nc'.format(DA_exp.file_name()))
else:
    B_std=np.empty((2,2))
    B_std[0,0]=np.std(B_ens.isel(lev=0,lev_d=0))
    B_std[0,1]=np.std(B_ens.isel(lev=0,lev_d=1))
    B_std[1,0]=B_std[0,1]
    B_std[1,1]=np.std(B_ens.isel(lev=1,lev_d=1))

    q_std=np.zeros((2,1))
    q_std[0]=np.std(q_full.isel(time=DA_days,lev=0))
    q_std[1]=np.std(q_full.isel(time=DA_days,lev=1))

    ml_std_ds=xr.Dataset({'B_std':xr.DataArray(B_std,coords=[mean_ds.lev,mean_ds.lev]),
                          'q_std':xr.DataArray(q_std.squeeze(),coords=[mean_ds.lev])})
    ml_std_ds.to_netcdf('./ML/{0}/std_{0}.nc'.format(DA_exp.file_name()))    
print(ml_std_ds)

### Process B data for training
B_stacked=B_ens.stack(sample=('time','y','x')).transpose('sample',...)
print(B_stacked.shape)

B_data=np.empty((len(B_ens.time)*len(B_ens.y)*len(B_ens.x),3,len(B_ens.y_d),len(B_ens.x_d)))
B_data[:,0,...]=B_stacked[:,0,0,...]/ml_std_ds.B_std[0,0].data
B_data[:,1,...]=B_stacked[:,0,1,...]/ml_std_ds.B_std[0,1].data
B_data[:,2,...]=B_stacked[:,1,1,...]/ml_std_ds.B_std[1,1].data
print(B_data.shape)

### Process q data for training
q_local=np.empty((len(q_full.time),len(q_full.lev),len(q_full.y),len(q_full.x),len(B_ens.y_d),len(B_ens.x_d)))
for i in range(len(q_full.x)):
    for j in range(len(q_full.y)):
        q_local[:,:,j,i,:,:]=DA.localize_q(q_full,j,i,DA_exp.Nx_DA,int(len(B_ens.x_d)/2))

q_local=q_local.transpose([0,2,3,1,4,5])
print(q_local.shape)
q_data=q_local.reshape((len(q_full.time)*len(q_full.y)*len(q_full.x),len(q_full.lev),len(B_ens.y_d),len(B_ens.x_d)))
print(q_data.shape)
q_data[:,0,...]=q_data[:,0,...]/ml_std_ds.q_std[0].data
q_data[:,1,...]=q_data[:,1,...]/ml_std_ds.q_std[1].data

q_unet=q_data[...,B_start:B_start+B_size,B_start:B_start+B_size]
B_unet=B_data[...,B_start:B_start+B_size,B_start:B_start+B_size]
B_shape=B_unet.shape
q_shape=q_unet.shape
print(B_shape,q_shape)

n_total=B_shape[0]
n_train=int(n_total*0.8)

train_ds=ML.Dataset(q_unet[0:n_train,...],B_unet[0:n_train,...],device)
valid_ds=ML.Dataset(q_unet[n_train:,...],B_unet[n_train:,...],device)

params = {'batch_size':16,'num_workers':1,'shuffle':True}
training_generator = torch.utils.data.DataLoader(train_ds, **params)
validation_generator = torch.utils.data.DataLoader(valid_ds, **params)

model=ML.Unet_2L(in_ch=len(in_ch),out_ch=len(out_ch))
model=model.to(device)
print(device)

# check keras-like model summary using torchsummary
# summary(model, input_size=q_shape[1:])

criterion = torch.nn.MSELoss() # MSE loss function
optimizer = optim.Adam(model.parameters(), lr=0.002)

model=model.double()
n_epochs = 2 #Number of epocs
validation_loss = list()
train_loss = list()
start_epoch=0
if start_epoch>0:
    model_file='./ML/unet_epoch{}_in{}_out{}_B{}_{}.pt'.format(start_epoch,''.join(map(str,in_ch)),''.join(map(str,out_ch)),B_size,DA_exp.file_name())
    print(model_file)
    model.load_state_dict(torch.load(model_file,map_location=torch.device('cpu')))
# time0 = time()  
for epoch in range(start_epoch+1, n_epochs + 1):
    train_loss.append(ML.train_model(model,criterion,training_generator,optimizer,device))
    # validation_loss.append(ML.test_model(model,criterion,validation_generator,optimizer,device))
    torch.save(model.state_dict(), './ML/{}/unet_epoch{}_in{}_out{}_B{}_{}.pt'.\
        format(DA_exp.file_name(),epoch,''.join(map(str,in_ch)),''.join(map(str,out_ch)),B_size,DA_exp.file_name()))

plt.plot(train_loss,'b', label='training loss');
plt.plot(validation_loss,'r', label='validation loss');

plt.legend();