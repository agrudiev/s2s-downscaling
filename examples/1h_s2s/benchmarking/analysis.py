import os
import h5py
import tomllib
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from evaluation.plots import plot_maps
from utils import create_folder

def load_data(filenames, ens_index=0, lead_time_index=0):
    data = {}
    for name, path in filenames.items():
        with h5py.File(path, "r") as f:
            precip = f["precip"]
            shape = precip.shape
            lons = f["longitude"][:]
            lats = f["latitude"][:]
            times = f["time"][:] if "time" in f else None

            if len(shape) == 5:
                # (ensemble, lead_time, time, lat, lon)
                if ens_index >= shape[0] or lead_time_index >= shape[1]:
                    raise IndexError(f"{name}: ens_index or lead_time_index out of bounds.")
                precip = precip[ens_index, lead_time_index]  # (time, lat, lon)
                lead_dim = True

            elif len(shape) == 4:
                # (ensemble, time, lat, lon)
                if ens_index >= shape[0]:
                    raise IndexError(f"{name}: ens_index out of bounds.")
                precip = precip[ens_index]  # (time, lat, lon)
                lead_dim = True

            elif len(shape) == 3:
                # (time, lat, lon)
                lead_dim = False

            else:
                raise ValueError(f"Unsupported shape {shape} in {name}")

            data[name] = {
                "precip": precip[:], 
                "lons": lons,
                "lats": lats,
                "times": times,
                "lead_time_dim": lead_dim
            }

    return data


def align_all_data(data, ref_model="CombiPrecip", models=None):
    if models is None:
        models = list(data.keys())

    aligned = {}
    ref_times = data[ref_model]["times"]
    ref_times = np.array(ref_times)
    lons = data[ref_model]["lons"]
    lats = data[ref_model]["lats"]

    for model in models:
        ds = data[model]
        precip_raw = ds["precip"]
        time_raw = ds["times"]
        time_raw = np.array(time_raw)

        while precip_raw.ndim > 3:
            precip_raw = precip_raw[0]

        aligned_precip = []
        last_frame = None

        for t in ref_times:
            candidates = np.where(time_raw <= t)[0]
            if len(candidates) > 0:
                idx = candidates[-1]
                last_frame = precip_raw[idx]
            aligned_precip.append(last_frame)

        aligned[model] = {
            "precip": np.array(aligned_precip),
            "times": ref_times,
            "lons": lons,
            "lats": lats,
        }

    return aligned


def make_plots(filenames):
    data = load_data(filenames)

    all_models = ["S2S", "S2S Interp", "Diffusion Spatial", "Diffusion Spatial Interp", "Diffusion Spatio-temporal", "CombiPrecip"]
    ref_model = "CombiPrecip"

    aligned_data = align_all_data(data, ref_model=ref_model, models=all_models)
    ref_times = aligned_data[ref_model]["times"]
    lons, lats = aligned_data[ref_model]["lons"], aligned_data[ref_model]["lats"]
    extent = (lons.min(), lons.max(), lats.min(), lats.max())

    figs_dir = os.path.join(os.path.dirname(__file__), "figs")
    save_dir = os.path.join(figs_dir, "maps_gif")
    create_folder(figs_dir)
    create_folder(save_dir)

    for i, t in enumerate(ref_times):
        arrays = []
        titles = []

        for model in all_models:
            frame = aligned_data[model]["precip"][i]
            if frame is None:
                print(f"[WARN] Frame missing for {model} at T={i} ({t}) — skipping frame")
                break
            arrays.append(frame)
            titles.append(model)

        if len(arrays) == len(all_models):
            fig, _ = plot_maps(arrays, titles, [extent] * len(arrays))
            fig.savefig(os.path.join(save_dir, f"maps_comparison_T{i:04d}.png"))
            plt.close(fig)
        else:
            print(f"[WARN] Skipped frame {i}: incomplete data.")


def main():

    # directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    base = dirs["main"]["base"]
    figs_dir = os.path.join(os.path.dirname(__file__), "figs")
    save_dir = os.path.join(figs_dir, "maps_gif")
    test_data_dir = os.path.join(base, dirs["subs"]["test_data"])
    filenames = {
        "S2S": os.path.join(base, dirs["subs"]["test_data"], "det_s2s_nearest_low-pass_1h.h5"),
        "S2S Interp": os.path.join(base, dirs["subs"]["test_data"], "det_s2s_nearest_low-pass_1h_interp.h5"),
        "Diffusion Spatial": os.path.join(base, dirs["subs"]["simulations"], "diffusion/light_longer_cli100_ens4.h5"),
        "Diffusion Spatial Interp": os.path.join(base, dirs["subs"]["simulations"], "diffusion/light_longer_cli100_ens4_1h_interp.h5"),
        "Diffusion Spatio-temporal": os.path.join(base, dirs["subs"]["simulations"], "diffusion/light_longer_st_cli100_ens4.h5"),
        "CombiPrecip": os.path.join(base, dirs["subs"]["test_data"], "cpc_condtday1h.h5"),
    }

    data_dir = os.path.join(os.path.dirname(__file__), "data", "maps")
    make_plots(filenames)

    png_files = sorted(glob(os.path.join(save_dir, "maps_comparison_T*.png")))
    gif_path = os.path.join(save_dir, "maps_comparison.gif")




if __name__ == "__main__":
    main()
