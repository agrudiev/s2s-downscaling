# here I'm using the result of the datasets produced by
# examples/cpc_era5_wrf/engineering/engineer_cpc.py for
# simplicity :D

import os, h5py, tomllib
import numpy as np
import xarray as xr

from utils import write_precip_to_h5

from configs.cpc import get_config


def generate_summer_range(time):
    # Generate 6-hour intervals within summer months, from the first to the last datetime in `time`
    start = time[0]
    end = time[-1]
    complete_range = np.arange(start, end + np.timedelta64(6, 'h'), np.timedelta64(6, 'h'))
    
    # Filter for dates in June, July, and August
    summer_range = [date for date in complete_range if date.astype('datetime64[M]').astype(int) % 12 + 1 in [6, 7, 8]]
    
    return summer_range

def aggregate_and_save(file_dir, pre_file_name, file_name):
    with h5py.File(os.path.join(file_dir, pre_file_name), 'r') as f:
        precip = f['precip'][...]
        lats = f['latitude'][:]
        lons = f['longitude'][:]
    
    times = xr.open_dataset(os.path.join(file_dir, pre_file_name))['time'].values
    times = np.sort(times)
    
    # Generate the full summer 6-hourly range
    summer_range = generate_summer_range(times)
    
    # Aggregate precipitation data following the summer range
    agg_precip = []
    agg_times = []
    for agg_time in summer_range:
        # Find indices in `times` that match the current 6-hour window
        time_indices = [i for i, t in enumerate(times) if agg_time - np.timedelta64(6, 'h') < t <= agg_time]
        
        # Check if the 6-hour window has complete data
        if len(time_indices) == 6:
            precip_sum = np.sum(precip[time_indices], axis=0)
            agg_precip.append(precip_sum)
            agg_times.append(agg_time)

    # Save combined train and validation data to a single file
    dims_dict = {
        "time": np.array(agg_times),
        "latitude": lats,
        "longitude": lons
    }
    write_precip_to_h5(
        dims_dict, np.array(agg_precip),
        os.path.join(file_dir, file_name)
    )


def main():
    # directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    base = dirs["main"]["base"]
    train_data_dir = os.path.join(base, dirs["subs"]["train"])
    validation_data_dir = os.path.join(base, dirs["subs"]["validation"])
    test_data_dir = os.path.join(base, dirs["subs"]["test"])
    
    # extra configurations
    config = get_config()
    cpc_preprocessed_file_name = config.cpc_preprocessed_file_name
    cpc_aggregated_file_name = config.cpc_aggregated_file_name
    
    # main calls
    aggregate_and_save(test_data_dir, cpc_preprocessed_file_name, cpc_aggregated_file_name)
    aggregate_and_save(validation_data_dir, cpc_preprocessed_file_name, cpc_aggregated_file_name)
    aggregate_and_save(train_data_dir, cpc_preprocessed_file_name, cpc_aggregated_file_name)


if __name__ == "__main__":
    main()
