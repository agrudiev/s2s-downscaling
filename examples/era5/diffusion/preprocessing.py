import os, h5py
import numpy as np

from evaluation.plots import plot_psds, CURVE_CMAP
from utils import get_spatial_lengths
from engineering.spectrum import get_1dpsd

def plot(
    era5_data, era5_lons, era5_lats,
    cpc_data, cpc_lons, cpc_lats,
    lowpass_data,
):
    # Get spatial lengths
    era5_x_length, era5_y_length = get_spatial_lengths(era5_lons, era5_lats)
    cpc_x_length, cpc_y_length = get_spatial_lengths(cpc_lons, cpc_lats)
    
    # Get lambda star
    wavelengths, psd = get_1dpsd(
        era5_data, era5_x_length, era5_y_length, 
        rotation_angle=0,
        data_std=0.31,
    )
    
    # Get PSD star
    Nx, Ny = len(cpc_lons), len(cpc_lats)
    
    lambda_star = 900
    psd_star = 1e-3
    sigma_star = np.sqrt(1e-3 * (Nx*Ny))
    print(f"sigma star = {sigma_star:.2e}")
    
    # Plot
    arrays = (era5_data, cpc_data, lowpass_data)
    labels = ("ERA5", "CPC", "ERA5 + nearest neighbors + low-pass")
    spatial_lengths = ((era5_x_length, era5_y_length), (cpc_x_length, cpc_y_length), (cpc_x_length, cpc_y_length))
    colors = (CURVE_CMAP(1), CURVE_CMAP(2), CURVE_CMAP(-1))
    fig, _ = plot_psds(
        arrays, labels, spatial_lengths,
        lambda_star=lambda_star,
        psd_star=psd_star,
        colors=colors,
        max_threshold=1e-2,
        min_threshold=1e-11,
        data_std=0.31,
        rotation_angle=np.pi/3,
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figs_dir = os.path.join(script_dir, "figs")
    fig.savefig(os.path.join(figs_dir, "sigma_star.png"))


def main():
    # Files
    test_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/test_data"
    cpc_file = os.path.join(test_dir, "cpc.h5")
    era5_file = os.path.join(test_dir, "era5.h5")
    lowpass_file = os.path.join(test_dir, "era5_nearest_low-pass.h5")
    
    # Read files
    with h5py.File(cpc_file, 'r') as f:
        cpc_data = f['precip'][...]
        cpc_lons = f['longitude'][:]
        cpc_lats = f['latitude'][:]
    with h5py.File(era5_file, 'r') as f:
        era5_data = f['precip'][...]
        era5_lons = f['longitude'][:]
        era5_lats = f['latitude'][:]
    with h5py.File(lowpass_file, 'r') as f:
        lowpass_data = f['precip'][...]
    
    # Plot the PSDs
    plot(
        era5_data, era5_lons, era5_lats, 
        cpc_data, cpc_lons, cpc_lats,
        lowpass_data,
    )
    
    
if __name__ == "__main__":
    main()