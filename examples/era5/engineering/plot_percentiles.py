import os, h5py
import numpy as np
import matplotlib.pyplot as plt

from configs.resampling import get_config

def plot(train_data_dir, cpc_file):
    # Read total precipitation in chuncks
    with h5py.File(os.path.join(train_data_dir, cpc_file), "r") as f:
        total_size = f["time"][:].shape[0]
    total_precip = np.zeros(total_size)
    CHUNK_SIZE = 1000
    start_idx = 0
    while start_idx < total_size:
        end_idx = start_idx + CHUNK_SIZE
        with h5py.File(os.path.join(train_data_dir, cpc_file), "r") as f:
            chunk = f["precip"][start_idx:end_idx,:,:]
        total_precip[start_idx:end_idx] = np.sum(chunk, axis=(1,2))
        start_idx = end_idx
    
    # Adjust to mm/h per grid
    with h5py.File(os.path.join(train_data_dir, cpc_file), "r") as f:
        Nx, Ny = f["precip"][0,:,:].shape
    total_precip = total_precip[~np.isnan(total_precip)]
    total_precip = total_precip / (Nx * Ny)
    
    # Get percentiles
    dp = 1
    percentiles = np.arange(0, 100+dp, dp)
    percentiles_precip = np.percentile(total_precip, percentiles)
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.bar(percentiles, percentiles_precip)
    plt.xlabel("Percentile")
    plt.ylabel("Total precipitation (mm/h)")
    plt.grid()
    ax.set_frame_on(False)
    ax.get_xaxis().set_tick_params(which='both', color='white')
    ax.get_yaxis().set_tick_params(which='both', color='white')
    script_dir = os.path.dirname(os.path.realpath(__file__))
    figs_dir = os.path.join(script_dir, "figs")
    fig.savefig(os.path.join(figs_dir, "percentiles.png"))
    
    
def main():
    config = get_config()
    train_data_dir = config.train_data_dir
    CPC_FILE = "cpc_resampled.h5"
    plot(train_data_dir, CPC_FILE)


if __name__ == '__main__':
    main()