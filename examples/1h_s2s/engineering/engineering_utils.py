import os, glob, h5py
import numpy as np
import xarray as xr
from tqdm import tqdm
from datetime import timedelta
from pyproj import Transformer
from engineering.regridding import interpolate_data, regularize_grid, cut_data


def transform_cpc_coordinates(xs, ys):
    # Transform coordinates from swiss coordinates to lat,lon
    transformer = Transformer.from_proj(2056, 4326, always_xy=True)
    xx, yy = np.meshgrid(xs, ys)
    lon_2d, lat_2d = transformer.transform(xx, yy)

    return lon_2d, lat_2d


def regrid_cpc(data, xs, ys, extent):
    # Transform cooridinates from CH1903 to WGS84
    lon_2d, lat_2d = transform_cpc_coordinates(xs, ys)
    
    # Create new regular lon,lat vectors and grid
    new_lon, new_lat, new_lon_2d, new_lat_2d = regularize_grid(lon_2d, lat_2d, xs.size, ys.size)
    
    # Interpolation of the irregular latlon to the regular latlon
    new_data = interpolate_data(data, lon_2d, lat_2d, new_lon_2d, new_lat_2d)
    
    # Cut data to the swiss region
    new_lon, new_lat, new_data = cut_data(new_lon, new_lat, new_data, extent)
    
    return new_lat, new_lon, new_data


def regrid_era5(raw_era5_lats, raw_era5_lons, raw_era5_data, new_lats, new_lons, method="linear"):
    # Create grid
    old_lon_2d, old_lat_2d = np.meshgrid(raw_era5_lons, raw_era5_lats)
    new_lon_2d, new_lat_2d = np.meshgrid(new_lons, new_lats)
    
    # Create 2D arrays to interpolate
    new_data = interpolate_data(
        raw_era5_data,
        old_lon_2d,
        old_lat_2d,
        new_lon_2d,
        new_lat_2d,
        method=method,
    )
    
    return new_data


def regrid_wrf(ds, new_lons, new_lats):
    # Get arrays from the datasets
    lons_2d = ds.XLONG.values[[0],:,:]
    lats_2d = ds.XLAT.values[[0],:,:]
    data = ds.PREC_ACC_NC.values
    
    # Get grid from reference latitude and longitudes
    new_lon_2d, new_lat_2d = np.meshgrid(new_lons, new_lats)
    
    # Interpolation of the irregular latlon to the regular latlon
    new_data = interpolate_data(
        data,
        lons_2d,
        lats_2d,
        new_lon_2d,
        new_lat_2d,
    )
    
    return ds.XTIME.values, new_data


def concat_cpc(data_dir, initial_date, final_date):
    # Get associated times array
    times = [initial_date + timedelta(days=n, hours=h) for n in range((final_date - initial_date).days + 1) for h in range(24)]

    # Create daytime array with only hours
    hours = np.array([t.hour for t in times])  # Extract only the hour part from the datetime
    daytime = hours.reshape(-1, 1)  # Reshape to have the correct shape (time x 1)
    
    # Get first dataset
    pattern = f"CPC{initial_date.strftime('%y%j')}"
    file_path = glob.glob(os.path.join(data_dir, pattern) + '*')[0]
    h5_file = h5py.File(file_path, 'r')
    data = h5_file['dataset1/data1/data'][...]
    h5_file.close()
    
    # Memory allocation for concatenated array
    data = np.empty((len(times), *data.shape), dtype=data.dtype)
    
    # Loop over all the files in the directory matching the given dates
    for d in tqdm(range(int((final_date - initial_date).days)+1), desc="Concatenating CPC files"):
        date = initial_date + timedelta(d)
        pattern = f"CPC{date.strftime('%y%j')}"
        for (h, file_path) in enumerate(sorted(glob.glob(os.path.join(data_dir, pattern) + '*'))):
            with h5py.File(file_path, 'r') as h5_file:
                data[24*d+h,:,:] = h5_file['dataset1/data1/data'][...]
    
    return times, daytime, data


def concat_wrf(data_dir, initial_date, final_date):
    datasets = []
    current_date = initial_date
    print("Concatenating WRF files...")
    num_iterations = ((final_date - initial_date).days+1)*24
    with tqdm(total=num_iterations, desc="Processing", unit="file") as pbar:
        while current_date <= final_date:
            for h in range(24):
                file = f"wrfout_d03_{current_date.strftime('%Y-%m-%d')}_{h:02d}:00:00"
                ds = xr.open_dataset(os.path.join(data_dir, file))[['PREC_ACC_NC']]
                datasets.append(ds)
                pbar.update(1)
            current_date += timedelta(days=1)
    return xr.concat(datasets, dim="Time")


def check_and_sort_times(times, daytime, data):
    # Check if the times array is sorted
    if not np.all(times[:-1] <= times[1:]):
        # Get the sorted indices
        sorted_indices = np.argsort(times)
        # Sort times and rearrange other arrays accordingly
        times = times[sorted_indices]
        daytime = daytime[sorted_indices]
        data = data[sorted_indices]
    return times, daytime, data


def split_date_range(start_date, end_date, chunk_size):
    date_ranges = []
    while start_date < end_date:
        chunk_end_date = min(start_date + timedelta(days=chunk_size - 1), end_date)
        date_ranges.append((start_date, chunk_end_date))
        start_date = chunk_end_date + timedelta(days=1)
    return date_ranges


def generate_date_list(start_date, end_date):
    date_list = []
    current_date = start_date
    
    while current_date <= end_date:
        date_list.append(current_date)
        current_date += timedelta(days=1)
    
    return date_list