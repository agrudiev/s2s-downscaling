import os, ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    base = "/work/FAC/FGSE/IDYST/tbeucler/downscaling"
    config.train_data_dir = os.path.join(base, "mlima/data/train_data")
    config.low_percentile = 40
    config.low_divisor = 4
    config.medium_percentile = 95
    config.medium_multiplier = 2
    config.high_percentile = 99
    config.high_multiplier = 5
    
    return config