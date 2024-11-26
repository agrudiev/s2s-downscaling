import os, ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    base = "/work/FAC/FGSE/IDYST/tbeucler/downscaling"
    config.test_data_dir = os.path.join(base, "mlima/data/test_data")
    
    return config
