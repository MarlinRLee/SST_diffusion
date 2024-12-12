import netCDF4 as nc
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
import os

data_dir = '/scratch.global/lee02328/noaa_sst_data/nc_data' 
file_names = [f'sst.day.mean.{year}.nc' for year in range(1981, 2025)]
output_file = 'processed_sst_data.npy'

def load_and_select_region(file_name, lat_min=16.25, lat_max=32.25, lon_min=262+80, lon_max=278+80):#lat_min=15, lat_max=33, lon_min=255, lon_max=295
    try:
        # Open dataset and select region
        ds = xr.open_dataset(os.path.join(data_dir, file_name))
        sst_data_array = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))['sst'].values
        return sst_data_array[::-1]  # Reverse for consistent orientation
    except FileNotFoundError:
        print(f"File not found: {file_name}")
        return None

def preprocess_frame(frame):
    """
    Preprocess a single SST frame by replacing NA values and creating a mask.
    
    Args:
        frame (numpy array): 2D SST frame with NA values.
    Returns:
        tuple: (processed frame, mask)
    """
    mask = ~np.isnan(frame)  # Create mask: True for valid (ocean) data, False for NA (land)
    processed_frame = frame.copy()
    processed_frame[~mask] = 0  # Replace NA values with 0
    return processed_frame, mask.astype(np.float32)

for i in range(6):
    base_lon = 262 + (i) * 16
    # Stack data and masks across years
    sst_data_list = []
    mask_list = []
    for file_name in file_names:
        data = load_and_select_region(file_name, lon_min = base_lon, lon_max = base_lon+16)
        if data is not None:
            frames_and_masks = [preprocess_frame(frame) for frame in data]
            sst_data_list.extend([fm[0] for fm in frames_and_masks])
            mask_list.extend([fm[1] for fm in frames_and_masks])

    # Convert lists to arrays and save
    if sst_data_list:
        sst_data = np.stack(sst_data_list, axis=0)  # (time, lat, lon)
        masks = np.stack(mask_list, axis=0)  # (time, lat, lon)
        np.save(f'data/processed_sst_data{i}.npy', sst_data)
        np.save(f'data/sst_masks{i}.npy', masks)
    else:
        print("No data processed.")