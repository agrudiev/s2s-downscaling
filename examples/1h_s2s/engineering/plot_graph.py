import os
import h5py
import tomllib
import numpy as np
from evaluation.plots import plot_maps, PRECIP_CMAP as cmap


def plot(test_data_dir, time, s2s_time):
    # --- Load CPC ---
    with h5py.File(os.path.join(test_data_dir, "cpc_condtday1h.h5"), "r") as f:
        cpc_data = f["precip"][time]  # (lat, lon)
        cpc_lons = f["longitude"][:]
        cpc_lats = f["latitude"][:]
    print("CPC shape:", cpc_data.shape)

    # --- Load raw S2S (1st member only) ---
    with h5py.File(os.path.join(test_data_dir, "det_s2s.h5"), "r") as f:
        s2s_data = f["precip"][0, s2s_time]  # (lat, lon)
        s2s_lons = f["longitude"][:]
        s2s_lats = f["latitude"][:]
    print("S2S shape:", s2s_data.shape)

    # --- Build shared extent using CPC grid ---
    cpc_extent = (cpc_lons[0], cpc_lons[-1], cpc_lats[0], cpc_lats[-1])
    extents = (cpc_extent, cpc_extent)  # cut S2S to CPC region in plot

    # --- Prepare arrays and titles ---
    arrays = (s2s_data, cpc_data)
    titles = ("S2S (original)", "CombiPrecip")

    # --- Plot ---
    fig, _ = plot_maps(arrays, titles, extents)
    figs_dir = os.path.join(test_data_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    fig.savefig(os.path.join(figs_dir, f"maps_s2s_vs_cpc.png"))
    print("Figure saved!")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    base = dirs["main"]["base"]
    test_data_dir = os.path.join(base, dirs["subs"]["test_data"])

    time = 41
    s2s_time = 8

    plot(test_data_dir, time, s2s_time)


if __name__ == "__main__":
    main()
