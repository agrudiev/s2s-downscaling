import os, h5py
import xarray as xr
import numpy as np

from evaluation.plots import plot_maps, plot_psds, CURVE_CMAP as cmap
from utils import get_spatial_lengths

from configs.analysis import get_config

def plot(test_data_dir, time):
    # Combi Precip
    with h5py.File(os.path.join(test_data_dir,  "cpc.h5"), "r") as f:
        cpc_data = f["precip"][:,:,:]
        cpc_lons = f["longitude"][:]
        cpc_lats = f["latitude"][:]
        
    # Raw ERA5
    with h5py.File(os.path.join(test_data_dir, "era5.h5"), "r") as f:
        era5_data = f["precip"][:,:,:]
        era5_lons = f["longitude"][:]
        era5_lats = f["latitude"][:]
    
    # Nearest neighbor interpolation + low-pass filter
    with h5py.File(os.path.join(test_data_dir, "era5_nearest.h5"), "r") as f:
        nearest_data = f["precip"][:,:,:]
        
    # Linear interpolation + low-pass filter
    with h5py.File(os.path.join(test_data_dir, "era5_nearest_low-pass.h5"), "r") as f:
        lowpass_data = f["precip"][:,:,:]
    
    times = xr.open_dataset(os.path.join(test_data_dir, "cpc.h5"), engine='h5netcdf').time.values
    
    # Plot maps
    print("Plotting time: ", times[time])
    arrays = (era5_data[time], nearest_data[time], lowpass_data[time], cpc_data[time])
    titles = ("ERA5 (original)", "ERA5 (nearest)", "ERA5 (nearest + low-pass)", "CombiPrecip")
    cpc_extent = (cpc_lons[0], cpc_lons[-1], cpc_lats[0], cpc_lats[-1])
    extents = (cpc_extent,) * 4 # here we're cuttign ERA5 on the plot
    fig, _ = plot_maps(arrays, titles, extents)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    maps_dir = os.path.join(script_dir, "figs")
    fig.savefig(os.path.join(maps_dir, "maps.png"))
    
    # Plot PSDs of ERA5 and CombiPrecip
    arrays = (era5_data, cpc_data)
    titles = ("ERA5", "CombiPrecip")
    era5_spatial_lengths = get_spatial_lengths(era5_lons, era5_lats)
    cpc_spatial_lengths = get_spatial_lengths(cpc_lons, cpc_lats)
    colors = (cmap(1), cmap(2))
    fig, _ = plot_psds(arrays, ("ERA5", "CombiPrecip"), (era5_spatial_lengths, cpc_spatial_lengths), colors=colors)
    fig.savefig(os.path.join(maps_dir, "psds_era5_cpc.png"))
    
    
    # Plot PSDs of ERA5, ERA5 nearest and Combiprecip
    arrays = (era5_data, nearest_data, cpc_data)
    titles = ("ERA5", "ERA5 + nearest neighbors", "CombiPrecip")
    spatial_lengths = (era5_spatial_lengths, cpc_spatial_lengths, cpc_spatial_lengths)
    colors = (cmap(1), cmap(6), cmap(2))
    fig, _ = plot_psds(arrays, titles, spatial_lengths, colors=colors, min_threshold=1e-10)
    fig.savefig(os.path.join(maps_dir, "psds_with_nearest.png"))
    
    # Low-pass filter
    arrays = (era5_data, lowpass_data, cpc_data)
    titles = ("ERA5", "ERA5 + low-pass", "CombiPrecip")
    spatial_lengths = (era5_spatial_lengths, cpc_spatial_lengths, cpc_spatial_lengths)
    colors = (cmap(1), cmap(6), cmap(2))
    fig, _ = plot_psds(arrays, titles, spatial_lengths, colors=colors, min_threshold=1e-10)
    fig.savefig(os.path.join(maps_dir, "psds_with_lowpass.png"))
    
    # Add white noise 1
    var = 1
    white_noise = np.random.rand(*cpc_data.shape) * var
    arrays = (
        lowpass_data + white_noise,
        cpc_data + white_noise,
    )
    titles = ("ERA5", "CombiPrecip")
    spatial_lengths = (cpc_spatial_lengths, cpc_spatial_lengths)
    colors = (cmap(6), cmap(2))
    fig, _ = plot_psds(arrays, titles, spatial_lengths, colors=colors, min_threshold=1e-10)
    fig.savefig(os.path.join(maps_dir, "psds_var0.png"))
    
    # Add white noise 1
    var = 10
    white_noise = np.random.rand(*cpc_data.shape) * var
    arrays = (
        lowpass_data + white_noise,
        cpc_data + white_noise,
    )
    titles = ("ERA5", "CombiPrecip")
    spatial_lengths = (cpc_spatial_lengths, cpc_spatial_lengths)
    colors = (cmap(6), cmap(2))
    fig, _ = plot_psds(arrays, titles, spatial_lengths, colors=colors, min_threshold=1e-10)
    fig.savefig(os.path.join(maps_dir, "psds_var1.png"))
    
    # Add white noise 2
    var = 100
    white_noise = np.random.rand(*cpc_data.shape) * var
    arrays = (
        lowpass_data + white_noise,
        cpc_data + white_noise,
    )
    titles = ("ERA5", "CombiPrecip")
    spatial_lengths = (cpc_spatial_lengths, cpc_spatial_lengths)
    colors = (cmap(6), cmap(2))
    fig, _ = plot_psds(arrays, titles, spatial_lengths, colors=colors, min_threshold=1e-10)
    fig.savefig(os.path.join(maps_dir, "psds_var2.png"))
    
    # Add white noise 2
    var = 2000
    white_noise = np.random.rand(*cpc_data.shape) * var
    arrays = (
        lowpass_data + white_noise,
        cpc_data + white_noise,
    )
    titles = ("ERA5", "CombiPrecip")
    spatial_lengths = (cpc_spatial_lengths, cpc_spatial_lengths)
    colors = (cmap(6), cmap(2))
    fig, _ = plot_psds(arrays, titles, spatial_lengths, colors=colors, min_threshold=1e-10)
    fig.savefig(os.path.join(maps_dir, "psds_var3.png"))
    
    print("Done!")
    
    
    
def main():
    config = get_config()
    test_data_dir = config.test_data_dir
    time = -10
    plot(test_data_dir, time)
    
if __name__ == "__main__":
    main()