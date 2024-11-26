import os, h5py, tomllib
import xarray as xr
import numpy as np

from evaluation.plots import plot_maps, plot_psds, CURVE_CMAP as cmap
from utils import get_spatial_lengths


def plot(test_data_dir, time):
    # Combi Precip
    with h5py.File(os.path.join(test_data_dir,  "cpc_6h.h5"), "r") as f:
        cpc_data = f["precip"][1:9,:,:]
        cpc_lons = f["longitude"][:]
        cpc_lats = f["latitude"][:]
    print("cpc_data.shape: ", cpc_data.shape)
        
    # Raw S2S
    with h5py.File(os.path.join(test_data_dir, "s2s.h5"), "r") as f:
        s2s_data = f["precip"][:,:,:]
        s2s_lons = f["longitude"][:]
        s2s_lats = f["latitude"][:]
    print("s2s_data.shape: ", s2s_data.shape)
    
    # Nearest neighbor interpolation + low-pass filter
    with h5py.File(os.path.join(test_data_dir, "s2s_nearest.h5"), "r") as f:
        nearest_data = f["precip"][:,:,:]
    print("nearest_data.shape: ", nearest_data.shape)
        
    # Linear interpolation + low-pass filter
    with h5py.File(os.path.join(test_data_dir, "s2s_nearest_low-pass.h5"), "r") as f:
        lowpass_data = f["precip"][:,:,:]
    print("lowpass_data.shape: ", lowpass_data.shape)
    
    times = xr.open_dataset(os.path.join(test_data_dir, "cpc_6h.h5"), engine='h5netcdf').time.values
    
    # Plot maps
    print("Plotting time: ", times[time])
    arrays = (s2s_data[time], nearest_data[time], lowpass_data[time], cpc_data[time])
    titles = ("S2S (original)", "S2S (nearest)", "S2S (nearest + low-pass)", "CombiPrecip")
    cpc_extent = (cpc_lons[0], cpc_lons[-1], cpc_lats[0], cpc_lats[-1])
    extents = (cpc_extent,) * 4 # here we're cuttign s2s on the plot
    fig, _ = plot_maps(arrays, titles, extents)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    figs_dir = os.path.join(script_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    fig.savefig(os.path.join(figs_dir, "maps.png"))
    
    # Plot PSDs of s2s and CombiPrecip
    arrays = (s2s_data, cpc_data)
    titles = ("S2S", "CombiPrecip")
    s2s_spatial_lengths = get_spatial_lengths(s2s_lons, s2s_lats)
    cpc_spatial_lengths = get_spatial_lengths(cpc_lons, cpc_lats)
    colors = (cmap(1), cmap(2))
    fig, _ = plot_psds(arrays, ("s2s", "CombiPrecip"), (s2s_spatial_lengths, cpc_spatial_lengths), colors=colors)
    fig.savefig(os.path.join(figs_dir, "psds_s2s_cpc.png"))
    
    
    # Plot PSDs of S2S, S2S nearest and Combiprecip
    arrays = (s2s_data, nearest_data, cpc_data)
    titles = ("S2S", "S2S + nearest neighbors", "CombiPrecip")
    spatial_lengths = (s2s_spatial_lengths, cpc_spatial_lengths, cpc_spatial_lengths)
    colors = (cmap(1), cmap(6), cmap(2))
    fig, _ = plot_psds(arrays, titles, spatial_lengths, colors=colors, min_threshold=1e-10)
    fig.savefig(os.path.join(figs_dir, "psds_with_nearest.png"))
    
    # Low-pass filter
    arrays = (s2s_data, lowpass_data, cpc_data)
    titles = ("S2S", "S2S + low-pass", "CombiPrecip")
    spatial_lengths = (s2s_spatial_lengths, cpc_spatial_lengths, cpc_spatial_lengths)
    colors = (cmap(1), cmap(6), cmap(2))
    fig, _ = plot_psds(arrays, titles, spatial_lengths, colors=colors, min_threshold=1e-10)
    fig.savefig(os.path.join(figs_dir, "psds_with_lowpass.png"))
    
    # Add white noise 0
    var = 1
    white_noise = np.random.rand(*cpc_data.shape) * var
    arrays = (
        lowpass_data + white_noise,
        cpc_data + white_noise,
    )
    titles = ("S2S", "CombiPrecip")
    spatial_lengths = (cpc_spatial_lengths, cpc_spatial_lengths)
    colors = (cmap(6), cmap(2))
    fig, _ = plot_psds(arrays, titles, spatial_lengths, colors=colors, min_threshold=1e-10)
    fig.savefig(os.path.join(figs_dir, "psds_var0.png"))
    
    # Add white noise 1
    var = 10
    white_noise = np.random.rand(*cpc_data.shape) * var
    arrays = (
        lowpass_data + white_noise,
        cpc_data + white_noise,
    )
    titles = ("S2S", "CombiPrecip")
    spatial_lengths = (cpc_spatial_lengths, cpc_spatial_lengths)
    colors = (cmap(6), cmap(2))
    fig, _ = plot_psds(arrays, titles, spatial_lengths, colors=colors, min_threshold=1e-10)
    fig.savefig(os.path.join(figs_dir, "psds_var1.png"))
    
    # Add white noise 2
    var = 100
    white_noise = np.random.rand(*cpc_data.shape) * var
    arrays = (
        lowpass_data + white_noise,
        cpc_data + white_noise,
    )
    titles = ("S2S", "CombiPrecip")
    spatial_lengths = (cpc_spatial_lengths, cpc_spatial_lengths)
    colors = (cmap(6), cmap(2))
    fig, _ = plot_psds(arrays, titles, spatial_lengths, colors=colors, min_threshold=1e-10)
    fig.savefig(os.path.join(figs_dir, "psds_var2.png"))
    
    # Add white noise 3
    var = 1000
    white_noise = np.random.rand(*cpc_data.shape) * var
    arrays = (
        lowpass_data + white_noise,
        cpc_data + white_noise,
    )
    titles = ("S2S", "CombiPrecip")
    spatial_lengths = (cpc_spatial_lengths, cpc_spatial_lengths)
    colors = (cmap(6), cmap(2))
    fig, _ = plot_psds(arrays, titles, spatial_lengths, colors=colors, min_threshold=1e-10)
    fig.savefig(os.path.join(figs_dir, "psds_var3.png"))
    
    print("Done!")
    
    
    
def main():
    # directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    base = dirs["main"]["base"]
    test_data_dir = os.path.join(base, dirs["subs"]["test"])
    
    # extra configurations
    time = 2
    
    # main call
    plot(test_data_dir, time)


if __name__ == "__main__":
    main()
