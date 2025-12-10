import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from ML_core import Unet,Unet_2L
import DA_core as DA
import importlib
import torch
from os.path import exists
from torchsummary import summary
DA_kwargs={}

importlib.reload(DA)

# Setting up the truth, obs, etc
DA_setup_paras={'nens':1,
                'Nx_DA':32,          
                'Nx_truth':32,          
                'obs_freq':10,          
                'obs_err':[1,-5,5,-7],          
                'nobs':[50,50]}
DA_setup=DA.DA_exp(**DA_setup_paras)

# ds_truth=DA_setup.read_truth(years=20)
try:
    # 1. ATTEMPT TO READ THE FILE (Line 23 from your traceback)
    ds_truth = DA_setup.read_truth(years=20)
    print("INFO: Successfully read existing 20-year truth file.")

except FileNotFoundError:
    # 2. IF THE FILE IS NOT FOUND, GENERATE IT
    print("WARNING: Truth file not found. Generating new 20-year truth data.")
    ds_truth = DA_setup.generate_truth(years=20)


# ds_truth.q.isel(time=7299,lev=0).plot(size=10)
patch_std=np.empty((2,31,31))
for i in range(8):
    for j in range(8):
        patch=ds_truth.q.isel(x=slice(i*4,i*4+4),y=slice(j*4,j*4+4))
        patch_std[:,i,j]=(patch.max(dim=('x','y'))-patch.min(dim=('x','y'))).mean('time').values

plt.contourf(patch_std[0,:,:])
plt.colorbar()

plt.contourf(patch_std[1,:,:])
plt.colorbar()

# Coarsen resolution
DA_setup.hires_to_lores(years=20)

q_lo_ds=DA_setup.read_truth(years=20,interp=True)
q_hi_ds=xr.open_dataset(q_lo_ds.attrs['source_file'])

select_day=1825
q_low=q_lo_ds.q.isel(time=select_day).squeeze('model')
q_hi=q_hi_ds.q.isel(time=select_day).squeeze('model')
fig, axes=plt.subplots(2,2,figsize=(10,8))
for j,q,s in zip(range(2),[q_hi,q_low],["High","Low"]):
    for i,clim in enumerate([4.0E-5,1.0E-6]):
        im=axes[i,j].contourf(q.x/1.0E3,q.y/1.0E3,q.isel(lev=i),cmap='bwr',extend='both',
                              levels=np.linspace(-clim,clim,21,endpoint=True))
        plt.colorbar(im,ax=axes[i,j])
        axes[i,j].set_title('{} Res (Lev {})'.format(s,i))

## Spin up the model for IC
q_init=DA_setup.ens_spinup(years=25)
q_init

## Generate "truth"/control simulation
try:
    ds_truth=DA_setup.read_truth(years=10)
except:
    ds_truth=DA_setup.generate_truth(years=10)
print(ds_truth)

## Generate or read observations
if exists(DA_setup.obs_name()):
    obs_ds=DA_setup.read_obs()
else:
    obs_ds=DA_setup.generate_obs(years=10)

print(obs_ds)

B_filename = '{}/B_d50_Nx{}_500years.nc'.format(DA.read_data_dir, DA_setup.Nx_DA)
if not exists(B_filename):
    print(f"WARNING: B matrix file '{B_filename}' not found. Generating now (500 years)...")
    ds_truth=DA_setup.generate_truth(years=500)
    DA.B_calculation_3DVar(Nx=DA_setup.Nx_DA, delta_days=50, years=500, coeff=1)
    print("B matrix generation complete.")
else:
    print(f"INFO: B matrix file '{B_filename}' already exists.")

## Visualize the generated observations
truth_file=obs_ds.attrs['truth_file']
truth_ds=xr.open_dataset(truth_file)

select_day=364
q=truth_ds.q.isel(time=select_day).squeeze('model')
obs_q=obs_ds.q.isel(day=select_day)
xi_q=obs_ds.xi.isel(day=select_day)
yi_q=obs_ds.yi.isel(day=select_day)
li_q=obs_ds.li.isel(day=select_day)
x_q=obs_ds.x.data[xi_q]
y_q=obs_ds.y.data[yi_q]
fig, axes=plt.subplots(1,2,figsize=(16,6))
for i in range(len(q.lev)):
    im=axes[i].contourf(q.x,q.y,q.isel(lev=i),cmap='bwr',levels=20)
    plt.colorbar(im,ax=axes[i])
    
    axes[i].scatter(x_q[li_q==i],y_q[li_q==i],c='k')
    # axes[i].set_title('q (lev {}, {})'.format(i,np.timedelta64(q.time.values,'D')))

## Generate "truth"/control simulation
ds_truth=DA_setup.generate_truth(years=100)
print(ds_truth)

# Calculate B matrix for 3DVar
B_ds=DA.B_calculation_3DVar(Nx=DA_setup.Nx_DA,delta_days=50,years=100,coeff=1)

select_x,select_y=np.array([20,20,20]),np.array([10,20,30])

fig, axes=plt.subplots(len(select_x),3,figsize=(15,4*len(select_x)))
for j,x,y in zip(range(len(select_x)),select_x,select_y):
    xy0=x+y*DA_setup.Nx_DA
    xy1=x+y*DA_setup.Nx_DA+DA_setup.Nx_DA*DA_setup.Nx_DA
    corr_xy0=B_ds.corr.data[xy0,:].reshape(2,DA_setup.Nx_DA,DA_setup.Nx_DA)
    corr_xy1=B_ds.corr.data[xy1,:].reshape(2,DA_setup.Nx_DA,DA_setup.Nx_DA)

    im=axes[j,0].contourf(B_ds.x/1.0E3,B_ds.y/1.0E3,corr_xy0[0,:,:],cmap='bwr',levels=np.linspace(-1.05,1.05,22,endpoint=True))
    plt.colorbar(im,ax=axes[j,0])
    axes[j,0].set_title('Lev 0-0 correlation (y={},x={})'.format(y,x))

    im=axes[j,1].contourf(B_ds.x/1.0E3,B_ds.y/1.0E3,corr_xy0[1,:,:],cmap='bwr',levels=np.linspace(-1.05,1.05,22,endpoint=True))
    plt.colorbar(im,ax=axes[j,1])
    axes[j,1].set_title('Lev 0-1 correlation (y={},x={})'.format(y,x))

    im=axes[j,2].contourf(B_ds.x/1.0E3,B_ds.y/1.0E3,corr_xy1[1,:,:],cmap='bwr',levels=np.linspace(-1.05,1.05,22,endpoint=True))
    plt.colorbar(im,ax=axes[j,2])
    axes[j,2].set_title('Lev 1-1 correlation (y={},x={})'.format(y,x))


for nens,relax in zip([20],[0.]):
    DA_paras={'nens':nens,
            'DA_method':'EnKF',
            'Nx_DA':32,
            'Nx_truth':128,
            'obs_freq':10,
            'obs_err':[1,-5,5,-7],
            'nobs':[50,50],
            'R_W':100,
            'DA_freq':10,
            'save_B':False,
            'inflate':[1,relax],
            'output_str':'',
            'delta_days': 50}
    DA_exp=DA.DA_exp(**DA_paras)
    DA_exp.run_exp(DA_days=365,DA_start=0,**DA_kwargs)

# Setting up DA experiment what was used to train Unet
DA_training_paras={'nens':320,
                   'DA_method':'EnKF',
                   'Nx_DA':64,
                   'Nx_truth':128,
                   'obs_freq':10,
                   'obs_err':[1,-5,5,-7],
                   'DA_freq':10,
                   'save_B':False,
                   'nobs':[50,50],
                   'R_W':100,
                   'inflate':[1,0.45],
                   'delta_days': 50}
DA_training=DA.DA_exp(**DA_training_paras)

# Read trained Unet and normalization factors (standard deviations)
in_ch=[0,1]
out_ch=[0,1,2]
epoch=49
R_training=12
R_DA=12
features=32
Ulevels=2
if Ulevels==3:
    model=Unet(in_ch=len(in_ch),out_ch=len(out_ch),features=features).double()
elif Ulevels==2:
    model=Unet_2L(in_ch=len(in_ch),out_ch=len(out_ch),features=features).double()
model_file='./ML/{}/{}L_{}f/unet_epoch{}_in{}_out{}_B{}_{}.pt'.format(
    DA_training.file_name(),Ulevels,features,epoch,''.join(map(str,in_ch)),''.join(map(str,out_ch)),R_training*2,DA_training.file_name())
# model_file='./ML/unet_in{}_out{}_{}.pt'.format(''.join(map(str,in_ch)),''.join(map(str,out_ch)),DA_training.file_name())
print(model_file)
model.load_state_dict(torch.load(model_file))
model.eval()
ml_std_ds=xr.open_dataset('./ML/{0}/std_{0}.nc'.format(DA_training.file_name()))

for R_W in [100]:
    for nens,relax in zip([1],[0.0]):
        DA_paras={'nens':nens,
                'DA_method':'UnetKF',
                'Nx_DA':64,
                'Nx_truth':128,
                'obs_freq':10,
                'obs_err':[1,-5,5,-7],
                'nobs':[50,50],
                'R_W':R_W,
                'DA_freq':10,
                'save_B':False,
                'inflate':[1,relax],
                'B_alpha':0.0,
                'R_training':R_training,
                'R_DA':R_DA,
                'training_exp':DA_training,
                'delta_days': 50}
        DA_exp=DA.DA_exp(**DA_paras)
        
        DA_kwargs['ml_model']=model
        DA_kwargs['ml_std_ds']=ml_std_ds
        DA_kwargs['output_str']=''
        DA_kwargs['output_str']='UnetKF_Nx{}_128_ens{}_{}L{}f'.format(DA_exp.Nx_DA,DA_training.nens,Ulevels,features)

        DA_exp.run_exp(DA_days=7300,DA_start=0,**DA_kwargs)


# Static localization weight matrix
Nx_DA=DA_setup.Nx_DA
R=75.0E3

W_ds=DA.Localize_weights(Nx=Nx_DA,R=R,save_netcdf=True)
# W_ds=xr.open_dataset('{}/W_Nx{}_L{}.nc'.format(data_dir,Nx_DA,int(R/1000)))

select_x,select_y,select_l=10,10,0
select_xy=select_x+select_y*Nx_DA+select_l*Nx_DA*Nx_DA
W_xy=W_ds.W.data[select_xy,:].reshape(2,Nx_DA,Nx_DA)

fig, axes=plt.subplots(1,2,figsize=(10,4))
for i in range(2):
    im=axes[i].contourf(W_ds.x/1.0E3,W_ds.y/1.0E3,W_xy[i,:,:],cmap='bwr',levels=np.linspace(-1.05,1.05,22,endpoint=True))
    plt.colorbar(im,ax=axes[i])
    axes[i].set_title('Lev {} Localization weight to (lev={},y={},x={})'.
                      format(i,select_l,select_y,select_x))

# DA.read_data_dir='/scratch/cimes/feiyul/PyQG/data/training'
# DA.save_data_dir='/scratch/cimes/feiyul/PyQG/data/training'
for R_W in [100]:
    for nens,relax in zip([80],[0.5]):
        DA_paras={'nens':nens,
                'DA_method':'EnKF',
                'Nx_DA':32,
                'Nx_truth':128,
                'obs_freq':10,
                'obs_err':[1,-5,5,-7],
                'nobs':[50,50],
                'R_W':R_W,
                'DA_freq':10,
                'save_B':True,
                'inflate':[1,relax],
                'output_str':'',
                'delta_days':50}
        DA_exp=DA.DA_exp(**DA_paras)
        DA_exp.run_exp(DA_days=3650,DA_start=0,**DA_kwargs)
