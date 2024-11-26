import os, ml_collections, numpy as np

def get_config(raw_data_dir):
    config = ml_collections.ConfigDict()
    config.storm_dates = (
        (np.datetime64('2018-06-11'), np.datetime64('2018-06-12')),
        (np.datetime64('2021-06-28'), np.datetime64('2021-06-29')),
    )
    config.lead_time_files = {
        "1 week": (
            os.path.join(raw_data_dir, "det_sfc_2018-06-04.nc"),
            os.path.join(raw_data_dir, "det_sfc_2021-06-21.nc"),
        ),
        "2 week": (
            os.path.join(raw_data_dir, "det_sfc_2018-05-28.nc"),
            os.path.join(raw_data_dir, "det_sfc_2021-06-14.nc"),
        ),
        "3 week": (
            os.path.join(raw_data_dir, "det_sfc_2018-05-21.nc"),
            os.path.join(raw_data_dir, "det_sfc_2021-06-07.nc"),
        ),
    }
    
    return config
