import os, ml_collections, datetime

def get_config():
    config = ml_collections.ConfigDict()
    base = "/work/FAC/FGSE/IDYST/tbeucler/downscaling"
    config.raw_data_dir = os.path.join(base, "raw_data/era5/precip/")
    config.train_data_dir = os.path.join(base, "mlima/data/train_data")
    config.trainig_years = [2018, 2019, 2020, 2021, 2022, 2023]
    config.training_months = [6, 7, 8]
    config.test_data_dir = os.path.join(base, "mlima/data/test_data")
    config.storm_dates = [
        (datetime.datetime(2018, 6, 11), datetime.datetime(2018, 6, 12)),
        (datetime.datetime(2021, 6, 28), datetime.datetime(2018, 6, 29)),
    ]
    config.storm_files = [
        os.path.join(config.raw_data_dir, "era5_tp_2018_06_11-13.nc"),
        os.path.join(config.raw_data_dir, "era5_tp_2021_06_28-30.nc"),
    ]
    
    return config
