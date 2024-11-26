import os
import h5py
import numpy as np

from utils import write_dataset

from configs.resampling import get_config

def resample(
    train_data_dir,
    low_percentile,
    low_divisor,
    medium_percentile,
    medium_multiplier,
    high_percentile,
    high_multiplier,
):
    # Read total precipitation in a single shot
    with h5py.File(os.path.join(train_data_dir, "cpc.h5"), "r") as f:
        total_precip = np.sum(f["precip"][:,:,:], axis=(1,2))
        Nx, Ny = f["precip"][0,:,:].shape

    # Adjust to mm/h per grid and take out NaNs
    idxs_without_nans = ~np.isnan(total_precip)
    total_precip = total_precip[idxs_without_nans]
    total_precip = total_precip / (Nx * Ny)

    # Get quantiles
    low_p = np.percentile(total_precip, low_percentile)
    print("Low percentile: ", low_p)
    medium_p = np.percentile(total_precip, medium_percentile)
    print("Medium percentile: ", medium_p)
    high_p = np.percentile(total_precip, high_percentile)
    print("High percentile: ", high_p)

    # Resample and modify images
    with h5py.File(os.path.join(train_data_dir, "cpc.h5"), "r") as f:
        data = f["precip"][:,:,:]
        lons = f["longitude"][:]
        lats = f["latitude"][:]
        times = f["time"][:]
        
    # Adjust
    data = data[idxs_without_nans]
    times = times[idxs_without_nans]
        
    # Mask data
    low_mask = total_precip <= low_p
    low_data = data[low_mask,:,:]
    low_times = times[low_mask]
    print("Low data shape: ", low_data.shape)
    medium_mask = (total_precip > medium_p) & (total_precip <= high_p)
    medium_data = data[medium_mask,:,:]
    medium_times = times[medium_mask]
    print("Medium data shape: ", medium_data.shape)
    high_mask = total_precip > high_p
    high_data = data[high_mask,:,:]
    high_times = times[high_mask]
    print("High data shape: ", high_data.shape)
    rest_mask = ~(low_mask | medium_mask | high_mask)
    rest_data = data[rest_mask,:,:]
    rest_times = times[rest_mask]
    print("Rest data shape: ", rest_data.shape)
    
    # Resample
    # low
    low_data_idxs = [i for i in range(low_data.shape[0])]
    low_data_idxs = np.random.choice(low_data_idxs, size=int(low_data.shape[0] / low_divisor))
    low_data = low_data[list(low_data_idxs),:,:]
    low_times = low_times[list(low_data_idxs)]
    print("Low data shape after resampling: ", low_data.shape)
    # medium
    medium_data_list = []
    medium_times_list = []
    for _ in range(medium_multiplier):
        medium_data_list.append(medium_data)
        medium_times_list.append(medium_times)
    medium_data = np.concatenate(medium_data_list, axis=0)
    medium_times = np.concatenate(medium_times_list, axis=0)
    print("Medium data shape after resampling: ", medium_data.shape)
    # high
    high_data_list = []
    high_times_list = []
    for _ in range(high_multiplier):
        high_data_list.append(high_data)
        high_times_list.append(high_times)
    high_data = np.concatenate(high_data_list, axis=0)
    high_times = np.concatenate(high_times_list, axis=0)
    print("High data shape after resampling: ", high_data.shape)
    
    # Aggregate and save    
    data = np.concatenate([low_data, medium_data, high_data, rest_data], axis=0)
    times = np.concatenate([low_times, medium_times, high_times, rest_times], axis=0)
    
    # Shuffle everything
    idxs = np.arange(times.shape[0])
    np.random.shuffle(idxs)
    times = times[list(idxs)]
    data = data[list(idxs),:,:]
    
    write_dataset(times, lats, lons, data, os.path.join(train_data_dir, "cpc_resampled.h5"))
    
    print("Done!")
        

def main():
    config = get_config()
    train_data_dir = config.train_data_dir
    low_percentile = config.low_percentile
    low_divisor = config.low_divisor
    medium_percentile = config.medium_percentile
    medium_multiplier = config.medium_multiplier
    high_percentile = config.high_percentile
    high_multiplier = config.high_multiplier
    
    resample(
        train_data_dir,
        low_percentile,
        low_divisor,
        medium_percentile,
        medium_multiplier,
        high_percentile,
        high_multiplier,
    )

if __name__ == '__main__':
    main()
