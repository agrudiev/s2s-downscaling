import os, ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    base_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima"
    config.prior_file_path = os.path.join(
        base_dir, 
        "data/test_data/era5_nearest_low-pass.h5"
    )
    config.save_dir = os.path.join(base_dir, "data/simulations/diffusion")
    config.clip_max = 40
    config.num_samples = 10
    
    return config
