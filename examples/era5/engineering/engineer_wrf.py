import os, h5py, datetime
import numpy as np
from utils import write_dataset
from engineering_utils import concat_wrf, regrid_wrf
from configs.wrf import get_config

def process_year(storm_dir, new_lons, new_lats, initial_date, final_date):
    # Read and plot original datasets
    wrf_ds = concat_wrf(storm_dir, initial_date, final_date)
    
    # Postprocessed WRF data
    times, wrf_data = regrid_wrf(wrf_ds, new_lons, new_lats)
    
    # Clip negative values to 0
    wrf_data[wrf_data < 0] = 0
    
    return times, wrf_data


def main():
    config = get_config()
    
    test_data_dir = config.test_data_dir
    storm_dirs = config.storm_dirs
    storm_dates = config.storm_dates
    output_dir = config.output_dir
    
    # Get lat/lon reference
    cpc_file = os.path.join(test_data_dir, "cpc.h5")
    with h5py.File(cpc_file, 'r') as h5_file:
        lons = h5_file['longitude'][:]
        lats = h5_file['latitude'][:]
    
    first = True
    for storm_dir, storm_date in zip(storm_dirs, storm_dates):
        if first:
            first = False
            times, data = process_year(storm_dir, lons, lats, storm_date, storm_date + datetime.timedelta(days=1))
        else:
            new_times, new_data = process_year(storm_dir, lons, lats, storm_date, storm_date + datetime.timedelta(days=1))
            data = np.concatenate([data, new_data], axis=0)
            times = np.concatenate([times, new_times], axis=0)
    
    # Create and save new dataset
    write_dataset(times, lats, lons, data, os.path.join(output_dir, "wrf.h5"))


if __name__ == '__main__':
    main()
