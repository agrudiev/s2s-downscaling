import os, ml_collections, datetime

def get_config():
    config = ml_collections.ConfigDict()
    base = "/work/FAC/FGSE/IDYST/tbeucler/downscaling"
    data_dir = os.path.join(base, "mlima/data/output_wrf")
    config.test_data_dir = os.path.join(base, "mlima/data/test_data")
    config.storm_dirs = [
        os.path.join(data_dir, "2018061100_analysis"),
        os.path.join(data_dir, "2021062800_analysis"),
    ]
    config.storm_dates = [
        datetime.datetime(2018, 6, 11),
        datetime.datetime(2021, 6, 28),
    ]
    config.output_dir = os.path.join(base, "mlima/data/generated_forecasts/wrf")
    
    return config
