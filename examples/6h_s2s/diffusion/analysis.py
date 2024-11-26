import os, h5py, tomllib
from swirl_dynamics.data.hdf5_utils import read_single_array
from evaluation.plots import plot_maps, plot_cdfs, plot_psds, plot_pp
from utils import get_spatial_lengths


def run_analysis(s2s_path, sim_path, cpc_path, time_idxs, num_samples):
    # Read extent from file
    with h5py.File(cpc_path, "r") as f:
        lons = f['longitude'][:]
        lats = f['latitude'][:]
    extent = (lons.min(), lons.max(), lats.min(), lats.max())
    
    # Read the dataset from the .hdf5 file
    images = read_single_array(sim_path, "precip")

    titles = [f"Sample #{i+1}" for i in range(num_samples)]
    extents = [extent] * num_samples
    for time_idx in time_idxs:
        # Get the samples at the specified time index
        samples = images[time_idx, :, :, :]

        # Initialize arrays, titles, and extents for plotting
        arrays = [samples[i] for i in range(num_samples)]

        # Plot maps
        fig, _ = plot_maps(arrays, titles, extents)
        script_dir = os.path.dirname(os.path.realpath(__file__))
        figs_dir = os.path.join(script_dir, "figs")
        os.makedirs(figs_dir, exist_ok=True)
        fig.savefig(os.path.join(figs_dir, f"ensembles_t{time_idx}.png"))
    
    # Plot CDFs
    arrays = [images[:, i] for i in range(num_samples)]
    fig, _ = plot_cdfs(arrays, titles)
    fig.savefig(os.path.join(figs_dir, "cdf.png"))
    
    # Plot PSDs
    spatial_lengths = get_spatial_lengths(lons, lats)
    spatial_lengths = [spatial_lengths] * num_samples
    fig, _ = plot_psds(arrays, titles, spatial_lengths)
    fig.savefig(os.path.join(figs_dir, "psd.png"))
    
    # Plot PP
    fig, _ = plot_pp(arrays, titles)
    fig.savefig(os.path.join(figs_dir, "pp.png"))
    

def main():
    # directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    base = dirs["main"]["base"]
    test_data_dir = os.path.join(base, dirs["subs"]["train_data_dir"])
    simulations_dir = os.path.join(base, dirs["subs"]["validation_data_dir"])
    
    # extra configurations
    sim_path = os.path.join(simulations_dir, "diffusion/light_cli100_ens4.h5")
    s2s_path = os.path.join(test_data_dir, "det_s2s_nearest.h5")
    cpc_path = os.path.join(test_data_dir,"cpc.h5")
    time_idxs = [i for i in range(8)]
    num_samples = 3

    # main call    
    run_analysis(s2s_path, sim_path, cpc_path, time_idxs, num_samples)

    
if __name__ == "__main__":
    main()
