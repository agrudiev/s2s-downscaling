import os, ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.low_percentile = 40
    config.low_divisor = 1
    config.medium_percentile = 95
    config.medium_multiplier = 1
    config.high_percentile = 99
    config.high_multiplier = 1
    
    return config
