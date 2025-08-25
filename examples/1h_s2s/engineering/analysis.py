import os, h5py, tomllib
import xarray as xr
import numpy as np

from evaluation.plots import plot_maps, plot_psds, CURVE_CMAP as cmap
from utils import get_spatial_lengths


def plot(test_data_dir, time, s2s_time):
    # Combi Precip
    with h5py.File(os.path.join(test_data_dir,  "cpc_condtday1h.h5"), "r") as f:
        cpc_data = f["precip"][:,:,:]
        cpc_lons = f["longitude"][:]
        cpc_lats = f["latitude"][:]
    print("cpc_data.shape: ", cpc_data.shape)
        
    # Raw S2S
    with h5py.File(os.path.join(test_data_dir, "det_s2s.h5"), "r") as f:
        s2s_data = f["precip"][:, :, :, :]
        s2s_data = s2s_data[0]  # (time, lat, lon)
        s2s_lons = f["longitude"][:]
        s2s_lats = f["latitude"][:]
    print("s2s_data.shape: ", s2s_data.shape)
    
    # Nearest neighbor interpolation + low-pass filter
    with h5py.File(os.path.join(test_data_dir, "det_s2s_nearest.h5"), "r") as f:
        nearest_data = f["precip"][:, :, :, :]
        nearest_data = nearest_data[0]  # (time, lat, lon)
    print("nearest_data.shape: ", nearest_data.shape)
        
    # Linear interpolation + low-pass filter
    with h5py.File(os.path.join(test_data_dir, "det_s2s_nearest_low-pass.h5"), "r") as f:
        lowpass_data = f["precip"][:, :, :, :]
        lowpass_data = lowpass_data[0]  # (time, lat, lon)
    print("lowpass_data.shape: ", lowpass_data.shape)

    # nearest_data = nearest_data[:8, :, :]
    # lowpass_data = lowpass_data[:8, :, :]
    
    times = xr.open_dataset(os.path.join(test_data_dir, "cpc_condtday1h.h5"), engine='h5netcdf').time.values
    
    # Plot maps
    print("Plotting time: ", times[time])
    arrays = (s2s_data[s2s_time], nearest_data[s2s_time], lowpass_data[s2s_time], cpc_data[time])
    titles = ("S2S (original)", "S2S (linear)", "S2S (linear + low-pass)", "CombiPrecip")
    cpc_extent = (cpc_lons[0], cpc_lons[-1], cpc_lats[0], cpc_lats[-1])
    extents = (cpc_extent, ) * 4                                          # here we're cuttign s2s on the plot
    fig, _ = plot_maps(arrays, titles, extents)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    figs_dir = os.path.join(script_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    fig.savefig(os.path.join(figs_dir, "maps.png"))
    


def main():
    # directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    base = dirs["main"]["base"]
    test_data_dir = os.path.join(base, dirs["subs"]["test_data"])
    
    # extra configurations
    time = 36
    s2s_time = 7
    
    # main call
    plot(test_data_dir, time, s2s_time)


if __name__ == "__main__":
    main()
