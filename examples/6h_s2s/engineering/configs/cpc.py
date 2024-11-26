import os, ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.cpc_preprocessed_file_name = "cpc.h5"
    config.cpc_aggregated_file_name = "cpc_6h.h5"
    
    return config
