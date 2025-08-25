import os, h5py, tomllib
import numpy as np
import xarray as xr

from engineering.regridding import interpolate_data
from engineering.spectrum import get_psd, apply_low_pass_filter2
from utils import write_precip_to_h5, get_spatial_lengths

from configs.s2s import get_config


def cut_det_s2s_data_and_coords(raw_s2s_data, raw_s2s_lats, raw_s2s_lons, cpc_lats, cpc_lons):
    min_lat, max_lat = cpc_lats[0], cpc_lats[-1]
    min_lon, max_lon = cpc_lons[0], cpc_lons[-1]
    lat_idxs = np.where((raw_s2s_lats >= min_lat) & (raw_s2s_lats <= max_lat))[0]
    lon_idxs = np.where((raw_s2s_lons >= min_lon) & (raw_s2s_lons <= max_lon))[0]
    cut_s2s_data = raw_s2s_data[:, :, lat_idxs[0]:lat_idxs[-1]+1, lon_idxs[0]:lon_idxs[-1]+1]
    s2s_lons = raw_s2s_lons[lon_idxs]
    s2s_lats = raw_s2s_lats[lat_idxs]
    
    return cut_s2s_data, s2s_lats, s2s_lons


def regrid_s2s(raw_lats, raw_lons, raw_data, new_lats, new_lons, method="linear"):
    old_lon_2d, old_lat_2d = np.meshgrid(raw_lons, raw_lats)
    new_lon_2d, new_lat_2d = np.meshgrid(new_lons, new_lats)
    
    new_data = []
    for member in range(raw_data.shape[0]):
        member_data = interpolate_data(
            raw_data[member],
            old_lon_2d,
            old_lat_2d,
            new_lon_2d,
            new_lat_2d,
            method=method,
        )
        new_data.append(member_data)
    
    new_data = np.stack(new_data, axis=0)
    
    return new_data


def get_idxs_to_keep(times, bound_dates):
    idxs_to_keep = []
    start, end = bound_dates
    start = np.datetime64(start) + np.timedelta64(6, 'h')
    end = np.datetime64(end) + np.timedelta64(24, 'h')
    idxs_to_keep = [i for i, t in enumerate(times) if start <= t <= end]
    return idxs_to_keep


def read_and_concatenate_s2s_precip(lead_time_files, storm_dates):
    lead_times = []
    times = []
    raw_s2s_lats = None
    aux_raw_s2s_data = None
    raw_s2s_lons = None
    raw_s2s_data = None
    
    first = True
    for lead_time, file_tuple in lead_time_files.items():
        lead_times.append(lead_time)
        
        for i, file in enumerate(file_tuple):
            ds = xr.open_dataset(file)
            idxs_to_keep = get_idxs_to_keep(ds.time.values, storm_dates[i])
            idxs_to_keep_diff = np.concatenate([np.array([idxs_to_keep[0]-1]), idxs_to_keep])
            precip = np.expand_dims(np.diff(ds.tp.values[idxs_to_keep_diff], axis=0), axis=0)
            
            if raw_s2s_lats is None:
                raw_s2s_lats = ds.latitude.values
                raw_s2s_lons = ds.longitude.values
                raw_s2s_data = precip
                times = ds.time.values[idxs_to_keep]
            
            elif first:
                raw_s2s_data = np.concatenate([raw_s2s_data, precip], axis=1)
                times = np.concatenate([times, ds.time.values[idxs_to_keep]])
            
            else:
                if aux_raw_s2s_data is None:
                    aux_raw_s2s_data = precip
                else:
                    aux_raw_s2s_data = np.concatenate([aux_raw_s2s_data, precip], axis=1)
        
        if not first:
            raw_s2s_data = np.concatenate([raw_s2s_data, aux_raw_s2s_data], axis=0)
            aux_raw_s2s_data = None
            
        first = False
    
    return lead_times, times, raw_s2s_lats, raw_s2s_lons, raw_s2s_data


def apply_low_pass_filter(nearest_s2s_data, cpc_lats, cpc_lons):
        x_length, y_length = get_spatial_lengths(cpc_lons, cpc_lats)
        s2s_x_length, s2s_y_length = get_spatial_lengths(cpc_lons, cpc_lats)
        nearest_lowpassed_s2s_data = []
        for lead_time in range(nearest_s2s_data.shape[0]):
            k, _ = get_psd(nearest_s2s_data[lead_time], s2s_x_length, s2s_y_length)
            cutoff = np.max(k)
            lowpassed_s2s_data = apply_low_pass_filter2(
                nearest_s2s_data[lead_time], cutoff, x_length, y_length,
            )
            nearest_lowpassed_s2s_data.append(lowpassed_s2s_data)
        nearest_lowpassed_s2s_data = np.stack(nearest_lowpassed_s2s_data, axis=0)
        return nearest_lowpassed_s2s_data
    
    
def process_test_data(test_data_dir, storm_dates, lead_time_files, hourly_resolution):
    # Read info from already preprocessed CPC data
    cpc_file = os.path.join(test_data_dir, f"cpc_condtday1h.h5")
    with h5py.File(cpc_file, "r") as f:
        cpc_lons = f["longitude"][:]
        cpc_lats = f["latitude"][:]
    
    # Aggregate S2S data
    lead_times, times, raw_s2s_lats, raw_s2s_lons, raw_s2s_data = read_and_concatenate_s2s_precip(
        lead_time_files, storm_dates,
    )
    
    # Sort data
    if raw_s2s_lats[0] > raw_s2s_lats[-1]:
        raw_s2s_lats = raw_s2s_lats[::-1]
        raw_s2s_data = raw_s2s_data[:, ::-1, :]
    if raw_s2s_lons[0] > raw_s2s_lons[-1]:
        raw_s2s_lons = raw_s2s_lons[::-1]
        raw_s2s_data = raw_s2s_data[:, :, ::-1]
        
    # Scale
    raw_s2s_data = raw_s2s_data / hourly_resolution
    
    # Cut according to lat/lon bounds
    cut_s2s_data, s2s_lats, s2s_lons = cut_det_s2s_data_and_coords(
        raw_s2s_data, raw_s2s_lats, raw_s2s_lons, cpc_lats, cpc_lons,
    )
    print(f"Cut S2S data shape: {cut_s2s_data.shape}")
    
    # Interpolate s2s to the same resolution as CombiPrecip
    nearest_s2s_data = regrid_s2s(
        raw_s2s_lats, raw_s2s_lons, raw_s2s_data, cpc_lats, cpc_lons, method='linear',
    )
    print(f"Nearest S2S data shape: {nearest_s2s_data.shape}")
    
    # (Radial) low-pass filter in each lead time
    nearest_lowpassed_s2s_data = apply_low_pass_filter(
        nearest_s2s_data, cpc_lats, cpc_lons,
    )
    print(f"Nearest low-passed S2S data shape: {nearest_lowpassed_s2s_data.shape}")
    
    # Create and save new datasets
    dims_dict = {
        "lead_time": np.array(lead_times),
        "time": times,
        "latitude": s2s_lats,
        "longitude": s2s_lons,
    }
    write_precip_to_h5(
        dims_dict, cut_s2s_data,
        os.path.join(test_data_dir, "det_s2s.h5"),
    )
    dims_dict = {
        "lead_time": np.array(lead_times),
        "time": times,
        "latitude": cpc_lats,
        "longitude": cpc_lons,
    }
    write_precip_to_h5(
        dims_dict, nearest_s2s_data,
        os.path.join(test_data_dir, "det_s2s_nearest.h5"),
    )
    write_precip_to_h5(
        dims_dict, nearest_lowpassed_s2s_data,
        os.path.join(test_data_dir, "det_s2s_nearest_low-pass.h5"),
    )
    
    
def main():
    # directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    base = dirs["main"]["base"]
    raw_data_dir = dirs["raw"]["s2s"]
    test_data_dir = os.path.join(base, dirs["subs"]["test_data"])
    
    # extra configurations
    config = get_config(raw_data_dir)
    storm_dates = config.storm_dates
    lead_time_files = config.lead_time_files
    hourly_resolution = 6
    
    # main call
    process_test_data(
        test_data_dir, storm_dates, lead_time_files, hourly_resolution,
    )


if __name__ == "__main__":
    main()
    print("Done!")
