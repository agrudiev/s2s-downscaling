import os, ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    base = "/work/FAC/FGSE/IDYST/tbeucler/downscaling"
    config.raw_data_dir = os.path.join(base, "MeteoSwiss_Products/CPCH")
    config.test_data_dir = os.path.join(base, "agrudiev/data/1h_s2s/test_data")
    config.train_data_dir = os.path.join(base, "agrudiev/data/1h_s2s/train_data")
    config.validation_data_dir = os.path.join(base, "agrudiev/data/1h_s2s/validation_data")
    config.extent = (2255000, 2965000, 840000, 1480000)  # xmin, xmax, ymin, ymax
    config.new_extent = (5.9, 10.6, 45.8, 47.9) # lon_min, lon_max, lat_min, lat_max
    config.num_workers = 5
    config.years = [2018, 2019, 2020, 2021, 2022, 2023]
    config.months = [6, 7, 8]
    config.validation_years = [2023]
    config.test_dates = {
        "2021": [(6, 28), (6, 29)],
        "2018": [(6, 11), (6, 12)]
    }
    
    return config