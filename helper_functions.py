import numpy as np
import xarray as xr
import calendar
from unet import UNet_nobatchnorm
import torch
import argparse

######### PARAMETERS ############
ssh_path_file = "" # Path to ssh data files
temp_path_file = "" # Path to temprature data files
vvel_path_file = "" # Path to meridional velocity data files
uvel_path_file = "" # Path to zonal velocity data files
#################################

# Argument parser
parser = argparse.ArgumentParser(description="Train U-Net model.")
parser.add_argument('--dummy', action='store_true', help="Use dummy data instead of real data")
args = parser.parse_args()

def list_files(uvel=False, vvel=False, temp=False, dummy=True):
    """
    This function returns one whole year (2007) of daily data as the training set 
    and 4 months (January, April, July, of next year (2008) as the testing set

    Options:

    infer zonal velocity field (uvel=True)
    infer meridional velocity field (vvel=True)
    infer temperature field (temp=True)

    The paths and file names should be adapted to the users dataset
    """

    train_files = []
    test_files = []

    if args.dummy:
        ssh_filename = f"./dummy_dataset/SSH_t.cfc11.20070101.nc"
        temp_filename = f"./dummy_dataset/TEMP_t.cfc11.20070101.nc"  
        vvel_filename = f"./dummy_dataset/VVEL_t.cfc11.20070101.nc"
                
        train_files.append(ssh_filename)
        train_files.append(temp_filename)
        train_files.append(vvel_filename)
        ssh_filename = f"./dummy_dataset/SSH_t.cfc11.20080101.nc"
        temp_filename = f"./dummy_dataset/TEMP_t.cfc11.20080101.nc"  
        vvel_filename = f"./dummy_dataset/VVEL_t.cfc11.20080101.nc"
                
        test_files.append(ssh_filename)
        test_files.append(temp_filename)
        test_files.append(vvel_filename)
    else:
        # TRAIN SET
        for year in range(2007, 2008):  # Covers 2007
            for month in range(1, 13):  # Months from 1 to 12
                num_days = calendar.monthrange(year, month)[1]
                for day in range(1, num_days + 1): # Looping up to the correct number of days in the month
                    day_str = f"{day:02d}"  # Format day as two digits (01, 02, ..., 31)
                    month_str = f"{month:02d}"  # Format month as two digits (01, 02, ..., 12)

                    ssh_filename = f"{ssh_path_file}SSH_t.cfc11.{year}{month_str}{day_str}.nc"
                    temp_filename = f"{temp_path_file}TEMP_t.cfc11.{year}{month_str}{day_str}.nc"

                    vvel_filename = f"{vvel_path_file}VVEL_t.cfc11.{year}{month_str}{day_str}.nc"
                    uvel_filename = f"{uvel_path_file}UVEL_t.cfc11.{year}{month_str}{day_str}.nc"

                    if uvel:
                        output_filename = uvel_filename  
                    elif vvel:
                        output_filename = vvel_filename
                    elif temp:
                        output_filename = temp_filename

                    train_files.append(ssh_filename)
                    train_files.append(temp_filename)
                    train_files.append(output_filename)

        # TEST SET
        for year in range(2008, 2009):
            for month in range(1, 13, 3):
                num_days = calendar.monthrange(year, month)[1]
                for day in range(1, num_days + 1):  # Looping up to the correct number of days in the month
                    day_str = f"{day:02d}"  # Format day as two digits (01, 02, ..., 31)
                    month_str = f"{month:02d}"  # Format month as two digits (01, 02, ..., 12)

                    ssh_filename = f"/work/uo0780/u302044/POPCFC/YEARLY/SSH/SSH_t.cfc11.{year}{month_str}{day_str}.nc"
                    temp_filename = f"/work/uo0780/u302044/POPCFC/YEARLY/TEMP/TEMP_t.cfc11.{year}{month_str}{day_str}.nc"
                    ehf_filename = f"/work/uo0780/u302044/POPCFC/YEARLY/EDDYFLUXES/UVPTEMPP_{year}{month_str}{day_str}.nc"
                    vvel_filename = f"/work/uo0780/u241194/u241194/POPCFC/DATANC/VVEL_t.cfc11.{year}{month_str}{day_str}.nc"
                    uvel_filename = f"/work/uo0780/u241194/u241194/POPCFC/DATANC/UVEL_t.cfc11.{year}{month_str}{day_str}.nc"
                    
                    if uvel:
                        output_filename = uvel_filename  
                    elif vvel:
                        output_filename = vvel_filename
                    elif temp:
                        output_filename = temp_filename
                    else:
                        output_filename = ehf_filename

                    test_files.append(ssh_filename)
                    test_files.append(temp_filename)
                    test_files.append(output_filename)
    return train_files, test_files

def load_data_from_nc_as_lists(vvel=False, uvel=False, temp=False):
    """
    Loading the train and test sets as lists. For each file in the train and test lists 
    (e.g., ssh_1.nc, temp_1.nc, edf_1.nc, ssh_2.nc, temp_2.nc, etc.), the values ssh, temp, and ehf          
    were appended in that specific order.
    """
    train_files, test_files = list_files(vvel=vvel, uvel=uvel, temp=temp)
    nctrains = [xr.open_dataset(f) for f in train_files]
    nctest = [xr.open_dataset(f) for f in test_files]
    return nctrains, nctest

# Load all data into CPU memory. 
def preload_data(nctrains, total_records, N_inp, N_out, var_input_names, depth_levels, var_output, min_height, max_height, min_width, max_width, num_recs=1, jump_first_depth_level=False):
    all_input_data = np.zeros((total_records, N_inp, max_height-min_height, max_width-min_width))*np.nan
    all_output_data = np.zeros((total_records, N_out, max_height-min_height, max_width-min_width))*np.nan
    current_index = 0
    # Assuming nctrains is a list of the .nc files: ['ssh_1.nc', 'sst_1.nc', 'edf_1.nc', 'ssh_2.nc', 'sst_2.nc', 'edf_2.nc']
    for ncindex in range(0, len(nctrains), 3):  # Increment by 3 to process every 3 files as a group
        # Get the 3 files (e.g., ssh, sst, edf) for this group
        ssh_data = nctrains[ncindex]      # First file of the group (ssh)
        temp_data = nctrains[ncindex + 1]  # Second file of the group (sst)
        output_data = nctrains[ncindex + 2]  # Third file of the group (edf)

        # Define the record slice for the current group
        rec_slice = slice(current_index, current_index + num_recs)

        # Process input variables (e.g., ssh, sst, edf)
        for ind, var_name in enumerate(var_input_names):
            # Read data from each file in the group for the corresponding variable
            data = ssh_data if ind == 0 else temp_data
            data_slice = data[var_name].isel(i_index=slice(min_width,max_width), j_index=slice(min_height,max_height)) if ind == 0 else data[var_name].isel(i_index=slice(min_width,max_width), j_index=slice(min_height,max_height))[0, :, :]
            land_sea_mask = np.isnan(data_slice) # Land Sea Mask

            data_slice = np.nan_to_num(data_slice, nan=0.0)
            # Get the actual dimensions of the data_slice
            slice_height, slice_width = data_slice.shape[-2], data_slice.shape[-1]
            
            # Assign the data to the all_input_data array
            all_input_data[rec_slice, ind, :slice_height, :slice_width] = data_slice

        # Process output variables (if any)
        for depth_level in range(depth_levels):
            var_depth_level = depth_level + 1 if jump_first_depth_level else depth_level
            data_slice = output_data[var_output].isel(i_index=slice(min_width,max_width), j_index=slice(min_height,max_height))[var_depth_level, :, :]
            # Get the actual dimensions of the data_slice
            slice_height, slice_width = data_slice.shape[-2], data_slice.shape[-1]
            
            mask_ehf = land_sea_mask | (data_slice == 99999)  # Mask for NaNs from SSH and 99999 in EDDYFLUXES
            
            data_slice = data_slice.where(~mask_ehf, 0)
            if var_output == "TEMP":
                data_slice = np.nan_to_num(data_slice, nan=0.0)
            # Assign the data to the all_output_data array
            all_output_data[rec_slice, depth_level, :slice_height, :slice_width] = data_slice

        # Update current_index after processing the group of 3 files
        current_index += num_recs

    return all_input_data, all_output_data

# pull a batch of data from preloaded memory. To be called during training
def loaddata_preloaded_train(index, batch_size, all_input_data, all_output_data, lim, width):
    rec_slice = slice(index, index + batch_size)
    yslice = slice(0, lim) #It's important to start from 0 here: in preload_data, we have padded the data in the last elements when the input data has smaller dimensions. 
    xslice = slice(0, width)
    # print('rec_slice is:')
    # print(rec_slice)
    # print('mean of squared values of loaded input data:')
    # print("{0:0.32f}".format(np.nanmean(all_input_data[rec_slice, :, yslice, xslice]**2)))
    return (all_input_data[rec_slice, :, yslice, xslice], 
            all_output_data[rec_slice, :, yslice, xslice])
    
#pull data from preloaded memory for testing. The difference from this and loaddata_preloaded_train is that
#we load test data as one single batch for testing. 
def loaddata_preloaded_test(all_input_data, all_output_data, lim, width):
    #rec_slice = slice(index, index + batch_size)
    yslice = slice(0, lim)
    xslice = slice(0, width)
    # print('rec_slice is:')
    # print(rec_slice)
    # print('mean of squared values of loaded input data:')
    # print("{0:0.32f}".format(np.nanmean(all_input_data[rec_slice, :, yslice, xslice]**2)))
    return (all_input_data[:, :, yslice, xslice], 
            all_output_data[:, :, yslice, xslice])

def convert_to_ij(lon_min, lon_max, lat_min, lat_max):
    # Load dataset
    path = "./dummy_dataset/" if args.dummy else vvel_path_file
    ds = xr.open_dataset(f"{path}VVEL_t.cfc11.20070101.nc")

    # Get the 2D coordinate arrays
    lon_2d = ds.U_LON_2D.values
    lat_2d = ds.U_LAT_2D.values

    # Initialize mask of all False
    mask = np.ones(lon_2d.shape, dtype=bool)
    
    # Apply bounding box mask
    mask &= (lon_2d >= lon_min) & (lon_2d <= lon_max)
    mask &= (lat_2d >= lat_min) & (lat_2d <= lat_max)

    # Find indices where mask is True
    indices = np.argwhere(mask)

    if indices.size == 0:
        raise ValueError("No grid points found within the given bounding box.")

    # Get bounding box of the selected indices
    i_min = np.min(indices[:, 0])
    i_max = np.max(indices[:, 0])
    j_min = np.min(indices[:, 1])
    j_max = np.max(indices[:, 1])

    return i_min, i_max, j_min, j_max

def nn_pred(checkpoint, N_inp, N_out, Nbase, day, min_width, max_width, min_height, max_height, depth_level=None, model=None):
    # Instantiate the model with the correct parameters
    model = UNet_nobatchnorm(N_inp, N_out, bilinear=True, Nbase=Nbase).cuda() if not model else model # Move to GPU


    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set model to evaluation mode

    ssh_file = f"/work/uo0780/u241194/u241194/POPCFC/DATANC/SSH_t.cfc11.{day}.nc"
    sst_file = f"/work/uo0780/u241194/u241194/POPCFC/DATANC/TEMP_t.cfc11.{day}.nc"

    # Open the SSH and SST data using xarray
    ssh_data = xr.open_dataset(ssh_file)
    sst_data = xr.open_dataset(sst_file)

    # Extract SSH and SST variables (replace with the actual variable names)
    ssh = ssh_data['SSH']
    sst = sst_data['TEMP']  # Assuming 'TEMP' contains the SST data

    # Select region
    ssh_slice = ssh.isel(i_index=slice(min_width, max_width), j_index=slice(min_height, max_height))
    sst_slice = sst.isel(i_index=slice(min_width, max_width), j_index=slice(min_height, max_height))[0, :, :]
    
    land_sea_mask = np.isnan(ssh_slice) # Land Sea Mask

    # Stack SSH and SST into one input tensor (2 channels)
    input_data = np.stack([ssh_slice.values, sst_slice.values], axis=0)
    
    land_sea_mask = np.isnan(ssh_slice) # Land Sea Mask
    
    # All NAN values to 0
    input_data = np.nan_to_num(input_data, nan=0.0)
    # Normalize the data if required by your model (depending on the training process)
    input_data = input_data - checkpoint['mean_input'][None, :, None, None]
    input_data=input_data/np.sqrt(checkpoint['var_input'][None, :, None, None])
    
    # Convert the input data to a PyTorch tensor and add batch dimension
    input_tensor = torch.tensor(input_data, dtype=torch.float32).cuda()  # Add batch dimension and move to GPU
    
    # Step 4: Make predictions using the model
    with torch.no_grad():
        prediction = model(input_tensor)  # Perform inference
    
    prediction = prediction.cpu() * np.sqrt(checkpoint['var_output'][None, :, None, None]) + checkpoint['mean_output'][None, :, None, None]
    
    predicted_output = prediction.squeeze().cpu().numpy()  # Remove batch dimension and move to CPU
    if not depth_level:
        #mask_ehf = land_sea_mask[None, :, :] | (predicted_output == 99999)  # Mask for NaNs from SSH and 99999 in EDDYFLUXES
        #predicted_output = np.where(~mask_ehf, predicted_output, np.nan)
        return predicted_output
    # Extract the desired depth level (the 0th depth level in this case)
    predicted_depth_level = predicted_output[depth_level, :, :] if N_out > 1 else predicted_output 

    mask_ehf = land_sea_mask | (predicted_depth_level == 99999)  # Mask for NaNs from SSH and 99999 in EDDYFLUXES

    predicted_depth_level = np.where(~mask_ehf, predicted_depth_level, np.nan)

    return predicted_depth_level
