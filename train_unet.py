print("U-Net Training started!")
# A code with the U-Net architecture and learning rate annealing. 
# The code trains the UNet and saves the checkpoint.

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import r2_score as R2
from copy import deepcopy
import utils
from unet import UNet_nobatchnorm
from scipy.stats import pearsonr
import helper_functions as hf
import os
import psutil

######## PARAMETERS ###########

maxEpochs = 10 # Training epochs (Set to 10 for testing)
Nbase = 64 # depth of the UNet

depth_levels = 10 # depth levels of the output data (output channels) 
model_checkpoint_file = './' # Path to save checkpoint of the U-Net

###############################


# select the region
lon_min = -100
lon_max =  100
lat_min = -50
lat_max =  50


var_ssh = 'SSH' # variable name on nc file for ssh 
var_temp = 'TEMP' # variable name on nc file for temperature (in our case sst is the depth level 1 of the temperature file)
var_output = 'VVEL' # variable name on nc file for meridional velocity field (change in case of infering another variable such as the zonal velocity for example)

###############################

def log_memory(msg=""):
    """
    function used to monitor the memory used. 
    In this script is used only before and after 
    loading the dataset for the training and testing
    """
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 3)  # in GB
    print(f"[MEMORY] {msg} -- {mem:.2f} GB used")

def run_model(var_input_names, var_output_names, save_fn_prefix, N_inp, N_out, mean_input, mean_output, var_input, var_output):
    #function to move data from CPU to GPU memory
    def totorch(x):
        if torch.cuda.is_available():
            return torch.tensor(x, dtype = torch.float).cuda()
        else:
            return torch.tensor(x, dtype = torch.float)
    #main definition of the UNet. This is a standard UNet unit with no batchnorm layers. Nbase controls how deep the network is. You can
    #investigate the effects of Nbase, or other parameters, such as the way pooling is done for the downsampling layers in UNet_nobatchnorm.
    if torch.cuda.is_available():
        model = UNet_nobatchnorm(N_inp, N_out, bilinear = True, Nbase = Nbase).cuda()
    else:
        model = UNet_nobatchnorm(N_inp, N_out, bilinear = True, Nbase = Nbase)
    input = torch.randn(1,N_inp,max_height-min_height,max_width-min_width).to(device) #a fake piece of data with the same sizes as a snapshot of the training data. This is just
    
    #to estimate the size of the UNet. 
    output = model(input)
    print('Model has ', utils.nparams(model)/1e6, ' million params')

    inp_test, out_test = hf.loaddata_preloaded_test(all_test_input, all_test_output, max_height-min_height, max_width-min_width)
    print('shapes of input and output TEST data:')
    print(inp_test.shape, out_test.shape)
    with torch.no_grad():
        inp_test = totorch(inp_test)

    Tcycle = 10 #Cycle for the learning rate annealing.
    criterion_train  = nn.L1Loss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr0, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5*100) #weight_decay could be adjusted

    r2_test = np.zeros(maxEpochs)
    train_loss_per_epoch = np.zeros(maxEpochs)
    epochmin = []
    maxr2l = []
    learn = np.zeros(maxEpochs)
    minloss = 1000
    maxR2 = -1000 #Just some unrealistically low value
    minlosscount = 0
    perm = False

    model_best = deepcopy(model)  # Initialize before the training loop 
    corrs = []
    pvals = []
    print('Starting training loop')
    for epoch in tqdm(range(maxEpochs)):
        #Set the learning rate annealing scheme 
        lr = utils.cosineSGDR(optim, epoch, T0=Tcycle, eta_min=0, eta_max=lr0, scheme = 'constant')  
        
        model.train()
        index_perm = np.arange(0, Ntrain, batch_size)
        
        if perm:
            index_perm = np.random.permutation(index_perm)
        epoch_losses = []
        for index in index_perm:
            inp, out = hf.loaddata_preloaded_train(index, batch_size, all_train_input, all_train_output, max_height-min_height, max_width-min_width)            
            inp, out = totorch(inp), totorch(out)
            out_mod = model(inp)
            loss = criterion_train(out.squeeze(), out_mod.squeeze())
            #Set gradient to zero
            optim.zero_grad()
            #Compute gradients       
            loss.backward()
            #Update parameters with new gradient
            optim.step()
            
            epoch_losses.append(loss.item())
            #Record train loss
            #scheduler.step()
        # Save average training loss for this epoch
        train_loss_per_epoch[epoch] = np.mean(epoch_losses)
        model.eval()
        with torch.no_grad():
            
            out_mod=model(inp_test)
            
            r2 = R2(out_test.flatten(), (out_mod).cpu().numpy().flatten())
            r2_test[epoch] = r2
            #record current best model and best predictions
            if maxR2 <  r2:
                maxR2 = r2
                epochmin.append(epoch)
                maxr2l.append(maxR2)                
                model_best = deepcopy(model)
                corrs = []
                pvals = []

                # Ensure both arrays are on CPU and in numpy format
                true_out = out_test
                pred_out = (out_mod).cpu().numpy()  # already .cpu().numpy() earlier

                for depth in range(true_out.shape[1]):
                    true_flat = true_out[:, depth, :, :].flatten()
                    pred_flat = pred_out[:, depth, :, :].flatten()
    
                    corr, pval = pearsonr(true_flat, pred_flat)
                    corrs.append(corr)
                    pvals.append(pval)

                corr, pval = pearsonr(out_test.flatten(), pred_out.flatten())
                print('R2:', r2, ' corr: ', corr, ' pval: ', pval)

    print('Training finished')
    model_best.eval()
    with torch.no_grad():
        model_best.to('cpu') 
        out_mod = model_best(inp_test.to('cpu')).detach().cpu().numpy()
        
    print('Corr of best model:', pearsonr(out_test.flatten(), out_mod.flatten())[0])

    Nx, Ny = out_test.shape[2:]; Nx, Ny

    #save trained model
    print(out_mod.shape, 'outout model shape')
    fstr = f'{save_fn_prefix}_{Nbase}_{lr0}_{batch_size}_{depth_levels}'
    PATH = model_checkpoint_file + f'Debug_Unet_{fstr}.pth'
    torch.save({
        'epoch': epochmin[-1] if epochmin else None,  # Best epoch
        'model_state_dict': model_best.state_dict(),
        'r2_test': r2_test,
        'train_loss': train_loss_per_epoch,
        'epochmin': epochmin,
        'maxr2l': maxr2l,
        'mean_input': mean_input,
        'mean_output': mean_output,
        'var_input': var_input,
        'var_output': var_output,
        'corrs': corrs,
        'pvals': pvals,
    }, PATH)

nctrains, nctest = hf.load_data_from_nc_as_lists(vvel=True)

# Count how many complete sets of three files are in the training data
Ntrain = len(nctrains) // 3
# Count how many complete sets of three files are in the testing data
Ntest = len(nctest) // 3


# Batch size calculation: the second biggest multiple of Ntrain is used 
# (in case having enough memory one could use batch_size=Ntrain)

multiples = [m for m in range(1, Ntrain+1) if Ntrain % m == 0]

if len(multiples) < 2:
    # If only one or zero multiples found, pick the biggest one or fallback to Ntrain
    batch_size = multiples[-1] if multiples else Ntrain
else:
    # Pick the second biggest multiple
    batch_size = multiples[-2]

lr0 = 0.005*10/batch_size #maximum magnitude of the learning rate. Roughly should scale inversely to batch_size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print ('Running on ', device)

# region index calculation
min_height, max_height, min_width, max_width = hf.convert_to_ij(lon_min, lon_max, lat_min, lat_max)

#names of input variables, specific to the .nc files this code works on.
var_names = [var_ssh, var_temp, var_output]
var_input_names = [var_ssh, var_temp]
var_output_names = [var_output]
N_inp = len(var_input_names) 
N_out = depth_levels

save_fn_prefix  = 'any_{}{}_{}_nobatchnorm'.format(var_ssh, var_temp, var_output)
    
print('number of training records:', Ntrain)
print('number of testing records:', Ntest)

log_memory("Before loading data")
all_train_input, all_train_output = hf.preload_data(nctrains, Ntrain, N_inp, N_out, var_input_names, depth_levels, var_output, min_height, max_height, min_width, max_width)
all_test_input, all_test_output = hf.preload_data(nctest, Ntest, N_inp, N_out, var_input_names, depth_levels, var_output, min_height, max_height, min_width, max_width)
log_memory("After loading data")
print(all_train_input.shape, all_train_output.shape)

# Data Normalization
print("Computing Mean and Variance for Data Normalization")
#Compute mean and variance for normalization
mean_input = np.nanmean(all_train_input, axis=(0, 2, 3))
mean_output = np.nanmean(all_train_output, axis=(0, 2, 3))

all_train_input=all_train_input-mean_input[None, :, None, None]
all_train_output=all_train_output-mean_output[None, :, None, None]
all_test_input=all_test_input-mean_input[None, :, None, None]
all_test_output=all_test_output-mean_output[None, :, None, None]

var_input = np.nanmean(all_train_input**2, axis=(0, 2, 3))
var_output = np.nanmean(all_train_output**2, axis=(0, 2, 3))

print("Mean and variance of all input data:")
print(mean_input,var_input)
print("Mean and variance of all output data:")
print(mean_output,var_output)

# Set very small variances to avoid division by zero
var_output = np.where(var_output == 0, 1e-9, var_output)

#Scale the data so that they have variance of 
all_train_input=all_train_input/np.sqrt(var_input[None, :, None, None])
all_train_output=all_train_output/np.sqrt(var_output[None, :, None, None])
all_test_input=all_test_input/np.sqrt(var_input[None, :, None, None])
all_test_output=all_test_output/np.sqrt(var_output[None, :, None, None])

run_model(var_input_names, var_output_names, save_fn_prefix, N_inp, N_out, mean_input, mean_output, var_input, var_output)
