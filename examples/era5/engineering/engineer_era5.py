import os, h5py
import numpy as np
import xarray as xr

from engineering.spectrum import get_1dpsd, get_2dpsd, rectangular_low_pass_filter, radial_low_pass_filter
from utils import write_dataset, get_spatial_lengths

from engineering_utils import regrid_era5, generate_date_list
from configs.era5 import get_config

M_TO_MM = 1000


def process_train_data(
    raw_data_dir,
    train_data_dir,
    trainig_years,
    training_months,
    test_data_dir,
    storm_dates,
):
    # Read and concatenate ERA5 data
    first = True
    for year in trainig_years:
        if year == 2019 or year == 2020:
            # I had a problem when downloading the data for these years
            continue
        for month in training_months:
            file = os.path.join(raw_data_dir, f"era5_tp_{year}_{month:02d}.nc")
            print("Reading file:", file)
            ds = xr.open_dataset(file)
            if first:
                first = False
                times = ds.time.values
                raw_era5_lats = ds.latitude.values
                raw_era5_lons = ds.longitude.values
                raw_era5_data = ds.tp.values
            else:
                raw_era5_data = np.concatenate([raw_era5_data, ds.tp.values], axis=0)
                times = np.concatenate([times, ds.time.values], axis=0)
    
    # Take out the storm dates
    for dates in storm_dates:
        for date in generate_date_list(*dates):
            date = np.datetime64(date)
            first_idx = np.where(times == date)[0][0]
            day_idxs = slice(first_idx, first_idx + 24)
            raw_era5_data = np.delete(raw_era5_data, day_idxs, axis=0)
            times = np.delete(times, day_idxs)
    
    # Sort data
    if raw_era5_lats[0] > raw_era5_lats[-1]:
        raw_era5_lats = raw_era5_lats[::-1]
        raw_era5_data = raw_era5_data[:, ::-1, :]
    if raw_era5_lons[0] > raw_era5_lons[-1]:
        raw_era5_lons = raw_era5_lons[::-1]
        raw_era5_data = raw_era5_data[:, :, ::-1]
        
    # Scale
    raw_era5_data = raw_era5_data * M_TO_MM
    
    # Read info from already preprocessed CPC data
    cpc_file = os.path.join(test_data_dir, "cpc.h5")
    with h5py.File(cpc_file, "r") as f:
        cpc_lons = f["longitude"][:]
        cpc_lats = f["latitude"][:]
    
    # Interpolate ERA5 to the same resolution as CombiPrecip
    nearest_era5_data = regrid_era5(raw_era5_lats, raw_era5_lons, raw_era5_data, cpc_lats, cpc_lons, method='nearest')
    
    # Low-pass filter
    raw_era5_x_length, raw_era5_y_length = get_spatial_lengths(raw_era5_lons, raw_era5_lats)
    (kx, ky), _ = get_2dpsd(raw_era5_data, raw_era5_x_length, raw_era5_y_length)
    cutoff = np.max(kx), np.max(ky)
    x_length, y_length = get_spatial_lengths(cpc_lons, cpc_lats)
    nearest_lowpassed_era5_data = rectangular_low_pass_filter(nearest_era5_data, cutoff, x_length, y_length)
    
    # Create and save new datasets
    write_dataset(times, raw_era5_lats, raw_era5_lons, raw_era5_data, os.path.join(train_data_dir, "era5.h5"))
    write_dataset(times, cpc_lats, cpc_lons, nearest_era5_data, os.path.join(train_data_dir, "era5_nearest.h5"))
    write_dataset(times, cpc_lats, cpc_lons, nearest_lowpassed_era5_data, os.path.join(train_data_dir, "era5_nearest_low-pass.h5"))


def process_test_data(
    test_data_dir,
    storm_files,
):
    # Read info from already preprocessed CPC data
    cpc_file = os.path.join(test_data_dir, "cpc.h5")
    with h5py.File(cpc_file, "r") as f:
        cpc_lons = f["longitude"][:]
        cpc_lats = f["latitude"][:]
    
    # Read and concatenate ERA5 data
    first = True
    idxs_to_keep = slice(0, 48)
    for file in storm_files:
        ds = xr.open_dataset(file)
        if first:
            first = False
            times = ds.time.values[idxs_to_keep]
            raw_era5_lats = ds.latitude.values
            raw_era5_lons = ds.longitude.values
            raw_era5_data = ds.tp.values[idxs_to_keep]
        else:
            raw_era5_data = np.concatenate([raw_era5_data, ds.tp.values[idxs_to_keep]], axis=0)
            times = np.concatenate([times, ds.time.values[idxs_to_keep]], axis=0)
    
    # Sort data
    if raw_era5_lats[0] > raw_era5_lats[-1]:
        raw_era5_lats = raw_era5_lats[::-1]
        raw_era5_data = raw_era5_data[:, ::-1, :]
    if raw_era5_lons[0] > raw_era5_lons[-1]:
        raw_era5_lons = raw_era5_lons[::-1]
        raw_era5_data = raw_era5_data[:, :, ::-1]
        
    # Scale
    raw_era5_data = raw_era5_data * M_TO_MM
    
    # Interpolate ERA5 to the same resolution as CombiPrecip
    nearest_era5_data = regrid_era5(
        raw_era5_lats, raw_era5_lons, raw_era5_data, cpc_lats, cpc_lons, method='nearest'
    )
    
    # (Radial) low-pass filter
    x_length, y_length = get_spatial_lengths(cpc_lons, cpc_lats)
    raw_era5_x_length, raw_era5_y_length = get_spatial_lengths(raw_era5_lons, raw_era5_lats)
    k, _ = get_1dpsd(raw_era5_data, raw_era5_x_length, raw_era5_y_length)
    cutoff = np.max(k)
    nearest_lowpassed_era5_data = radial_low_pass_filter(
        nearest_era5_data, cutoff, x_length, y_length,
    )
    
    # Create and save new datasets
    write_dataset(
        times, raw_era5_lats, raw_era5_lons, raw_era5_data,
        os.path.join(test_data_dir, "era5.h5"),
    )
    write_dataset(
        times, cpc_lats, cpc_lons, nearest_era5_data,
        os.path.join(test_data_dir, "era5_nearest.h5"),
    )
    write_dataset(
        times, cpc_lats, cpc_lons, nearest_lowpassed_era5_data,
        os.path.join(test_data_dir, "era5_nearest_low-pass.h5"),
    )
    
    
def main():
    config = get_config()
    raw_data_dir = config.raw_data_dir
    train_data_dir = config.train_data_dir
    trainig_years = config.trainig_years
    training_months = config.training_months
    test_data_dir = config.test_data_dir
    storm_dates = config.storm_dates
    storm_files = config.storm_files
    
    process_train_data(
        raw_data_dir,
        train_data_dir,
        trainig_years,
        training_months,
        test_data_dir,
        storm_dates,
    )
    
    process_test_data(
        test_data_dir,
        storm_files,
    )


if __name__ == "__main__":
    main()
