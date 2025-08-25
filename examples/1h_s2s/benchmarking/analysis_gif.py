import os
import h5py
import tomllib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils import get_spatial_lengths, create_folder
from evaluation.plots import plot_maps, PRECIP_CMAP as cmap

def parse_precip_array(array, dimension_order, lead_time_idx=0, ensemble_idx=0):
    dim_map = {name: i for i, name in enumerate(dimension_order)}
    if "ensemble" in dim_map:
        array = array[ensemble_idx]
    if "lead_time" in dim_map:
        array = array[lead_time_idx]
    return array

def load_data(filenames, dimension_orders):
    data = {}
    for name, path in filenames.items():
        with h5py.File(path, "r") as f:
            precip_raw = f["precip"]
            lons = f["longitude"][:]
            lats = f["latitude"][:]
            times = f["time"][:] if "time" in f else None
            shape = precip_raw.shape

            dimension_order = dimension_orders.get(name)
            if not dimension_order:
                raise ValueError(f"Missing dimension order for dataset '{name}'")

            precip = parse_precip_array(precip_raw[:], dimension_order)

            data[name] = {
                "precip": precip,
                "lons": lons,
                "lats": lats,
                "times": times,
                "lead_time_dim": "lead_time" in dimension_order
            }

    return data

def get_common_times(data, ref="CombiPrecip", subset_models=None):
    if subset_models is None:
        subset_models = list(data.keys())

    common = set(data[ref]["times"])
    for k in subset_models:
        entry = data[k]
        if entry["times"] is None:
            raise ValueError(f"Missing time info for {k}")
        common &= set(entry["times"])
    common = sorted(common)

    print(f"[INFO] Timestamps in {ref}: {len(data[ref]['times'])}")
    for k in subset_models:
        print(f"[INFO] Timestamps in {k}: {len(data[k]['times'])}")
    print(f"[INFO] Common timestamps for selected datasets: {len(common)}")

    aligned = {}
    for k in subset_models + [ref]:
        d = data[k]
        idx = [i for i, t in enumerate(d["times"]) if t in common]
        precip = d["precip"][:, idx] if d["lead_time_dim"] else d["precip"][idx]
        aligned[k] = {
            "precip": precip,
            "lons": d["lons"],
            "lats": d["lats"],
            "times": [d["times"][i] for i in idx],
            "lead_time_dim": d["lead_time_dim"]
        }

    return aligned, common

def make_plots(filenames, dimension_orders):
    data = load_data(filenames, dimension_orders)

    long_models = ["S2S", "Diffusion"]
    short_models = ["S2S_interp", "DiffusionInterp", "DiffusionST"]
    ref_model = "CombiPrecip"

    aligned_long, times_long = get_common_times(data, ref=ref_model, subset_models=long_models)
    aligned_short, times_short = get_common_times(data, ref=ref_model, subset_models=short_models)

    lons, lats = aligned_long[ref_model]["lons"], aligned_long[ref_model]["lats"]
    extent = (lons.min(), lons.max(), lats.min(), lats.max())

    figs_dir = os.path.join(os.path.dirname(__file__), "figs/maps_gif")
    create_folder(figs_dir)

    ref_times = aligned_long[ref_model]["times"]
    max_time_idx = len(ref_times)

    for i in range(max_time_idx):
        arrays, titles = [], []

        for model in long_models:
            d = aligned_long[model]
            precip = d["precip"][0] if d["lead_time_dim"] else d["precip"]
            times = d["times"]
            idx = max([j for j, t in enumerate(times) if t <= ref_times[i]], default=None)
            if idx is not None and idx < len(precip):
                arrays.append(precip[idx])
                titles.append(f"{model} (6h)")

        for model in short_models:
            d = aligned_short[model]
            precip = d["precip"][0, i] if d["lead_time_dim"] else d["precip"][i]
            arrays.append(precip)
            titles.append(f"{model} (1h)")

        ref_precip = aligned_long[ref_model]["precip"][i]
        arrays.append(ref_precip)
        titles.append("CombiPrecip")

        labels = list(filenames)
        colors = [cmap(i) for i in range(len(labels))]

        if len(arrays) == 6:
            fig, _ = plot_maps(
                arrays,
                titles,
                [extent] * 6,
                cmap=colors,
                dpi=300
            )
            path = os.path.join(figs_dir, f"maps_comparison_T{i:04d}.png")
            fig.savefig(path)
            plt.close(fig)
        else:
            print(f"[WARN] Skipping T={i} — only {len(arrays)} arrays available.")

    print("Generating GIF...")
    images = []
    for i in range(max_time_idx):
        path = os.path.join(figs_dir, f"maps_comparison_T{i:04d}.png")
        if os.path.exists(path):
            images.append(Image.open(path))
    gif_path = os.path.join(figs_dir, "precip_evolution.gif")
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=200, loop=0)

def main():
    with open(os.path.join(os.path.dirname(__file__), "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    base = dirs["main"]["base"]

    filenames = {
        "S2S": os.path.join(base, dirs["subs"]["test_data"], "det_s2s_nearest_low-pass_1h.h5"),
        "S2S_interp": os.path.join(base, dirs["subs"]["test_data"], "det_s2s_nearest_low-pass_1h_interp.h5"),
        "Diffusion": os.path.join(base, dirs["subs"]["simulations"], "diffusion/light_longer_cli100_ens4.h5"),
        "DiffusionInterp": os.path.join(base, dirs["subs"]["simulations"], "diffusion/light_longer_cli100_ens4_1h_interp.h5"),
        "DiffusionST": os.path.join(base, dirs["subs"]["simulations"], "diffusion/light_longer_st_cli100_ens4.h5"),
        "CombiPrecip": os.path.join(base, dirs["subs"]["test_data"], "cpc_condtday1h.h5"),
    }

    dimension_orders = {
        "S2S": ("lead_time", "time", "lat", "lon"),
        "S2S_interp": ("lead_time", "time", "lat", "lon"),
        "Diffusion": ("ensemble", "lead_time", "time", "lat", "lon"),
        "DiffusionInterp": ("ensemble", "lead_time", "time", "lat", "lon"),
        "DiffusionST": ("ensemble", "lead_time", "time", "lat", "lon"),
        "CombiPrecip": ("time", "lat", "lon"),
    }

    make_plots(filenames, dimension_orders)

if __name__ == "__main__":
    main()
