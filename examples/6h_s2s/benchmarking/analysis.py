import os, h5py, tomllib
import xarray as xr

from evaluation.plots import *
from evaluation.metrics import *
from utils import create_folder, get_spatial_lengths


def load_data(filenames):
    data = {}
    for name, file in filenames.items():
        if name == "Diffusion" or name == "QM+Diffusion":
            with h5py.File(file, "r") as f:
                data[name] = {
                    "precip": f["precip"][:, 0, :, :],
                    "lons": f["longitude"][:],
                    "lats": f["latitude"][:],
                    "times": xr.open_dataset(file).time.values
                }
        else:
            with h5py.File(file, "r") as f:
                data[name] = {
                    "precip": f["precip"][:, :, :],
                    "lons": f["longitude"][:],
                    "lats": f["latitude"][:],
                    "times": xr.open_dataset(file).time.values
                }
    return data


def create_forecast_dicts(data, x_length, y_length):
    return [
        {"name": name, "data": values['precip'], "x_length": x_length, "y_length": y_length}
        for name, values in data.items()
    ]


def make_plots(filenames, time_idxs, colors, ls):
    # Load and preprocess data
    data = load_data(filenames)

    # Extract spatial extents
    lons, lats = data['CombiPrecip']['lons'], data['CombiPrecip']['lats']
    extent = (lons.min(), lons.max(), lats.min(), lats.max())
    x_length, y_length = get_spatial_lengths(lons, lats)

    # Plotting
    script_dir = os.path.dirname(os.path.realpath(__file__))
    figs_dir = os.path.join(script_dir, "figs")
    create_folder(figs_dir)

    # Plot maps
    first = True
    for time_idx in time_idxs:
        arrays = [data[key]['precip'][time_idx, :, :] for key in filenames.keys()]
        
        if first:
            print("Check times:")
            time_arrays = [data[key]['times'] for key in filenames.keys()]
            for times in time_arrays:
                print(times[-1])
            first = False
        
        titles = list(filenames.keys())
        extents = [extent] * len(arrays)
        fig, _ = plot_maps(arrays, titles, extents)
        gif_dir = os.path.join(figs_dir, "maps_gif")
        os.makedirs(gif_dir, exist_ok=True)
        fig.savefig(os.path.join(gif_dir, "maps_comparison_H{:02d}.png".format(time_idx)))

    # Plot CDFs
    arrays = [data[key]['precip'] for key in filenames.keys()]
    labels = list(filenames.keys())
    fig, _ = plot_cdfs(arrays, labels, colors=colors, ls=ls)
    fig.savefig(os.path.join(figs_dir, "cdf_comparison.png"))

    # Plot PSDs
    spatial_lengths = [(x_length, y_length) for _ in range(len(arrays))]
    fig, _ = plot_psds(arrays, labels, spatial_lengths, min_threshold=1e-10, colors=colors, ls=ls, lambda_star=680)
    fig.savefig(os.path.join(figs_dir, "psd_comparison.png"))
    
    # Plot PP
    fig, _ = plot_pp(arrays, labels, colors=colors, ls=ls)
    fig.savefig(os.path.join(figs_dir, "pp_comparison.png"))
    
    
def print_metrics(filenames):
    # Wasserstein distance
    data = load_data(filenames)
    
    # Extract spatial extents
    lons, lats = data['CombiPrecip']['lons'], data['CombiPrecip']['lats']
    x_length, y_length = get_spatial_lengths(lons, lats)
    

def main():
    # directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    base = dirs["main"]["base"]
    test_data_dir = os.path.join(base, dirs["subs"]["test"])
    simulations_dir = os.path.join(base, dirs["subs"]["simulations"])
    
    # extra configurations
    filenames = {
        "S2S": os.path.join(test_data_dir, "s2s_nearest.h5"),
        "Diffusion": os.path.join(simulations_dir, "diffusion/light_cli100_ens4.h5"),
        "CombiPrecip": os.path.join(test_data_dir, "cpc_6h.h5"),
    }
    time_idxs = range(0, 8)
    cmap = CURVE_CMAP
    colors = (
        cmap(0),
        cmap(6),
        cmap(2),
    )
    ls = (
        '-',
        '-',
        '-',
    )

    # main calls
    make_plots(filenames, time_idxs, colors, ls)
    print_metrics(filenames)


if __name__ == "__main__":
    main()
