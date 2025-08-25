import os
import h5py
import tomllib

import matplotlib.pyplot as plt

from evaluation.plots import *
from evaluation.metrics import *
from utils import create_folder, get_spatial_lengths


def load_data(filenames):
    data = {}
    for name, path in filenames.items():
        with h5py.File(path, "r") as f:
            precip = f["precip"]
            shape = precip.shape
            lons, lats = f["longitude"][:], f["latitude"][:]
            times = f["time"][:] if "time" in f else None

            if len(shape) == 5:
                precip = precip[:, 0]
                lead_dim = True
            elif len(shape) == 4:
                lead_dim = True
            elif len(shape) == 3:
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

    return aligned


def make_logpdf_plots(filenames, colors, ls):
    data = load_data(filenames)

    long_models = ["S2S", "Diffusion Spatial"]
    short_models = ["S2S Interpolated", "Diffusion Spatial Interp", "Diffusion Spatio-temporal"]
    ref_model = "CombiPrecip"

    aligned_long = get_common_times(data, ref=ref_model, subset_models=long_models)
    aligned_short = get_common_times(data, ref=ref_model, subset_models=short_models)

    lons, lats = aligned_long[ref_model]["lons"], aligned_long[ref_model]["lats"]
    xlen, ylen = get_spatial_lengths(lons, lats)

    figs_dir = os.path.join(os.path.dirname(__file__), "figs")
    create_folder(figs_dir)

    for lead in range(3):
        arrays, labels_plot, lengths = [], [], []

        for aligned in [aligned_long, aligned_short]:
            for model, d in aligned.items():
                if model == ref_model:
                    continue
                if d["lead_time_dim"]:
                    if lead >= d["precip"].shape[0]:
                        continue
                    arr = d["precip"][lead]
                else:
                    if lead > 0:
                        continue
                    arr = d["precip"]
                arrays.append(arr)
                labels_plot.append(model)
                lengths.append((xlen, ylen))

        obs_arr = aligned_long[ref_model]["precip"]
        arrays.append(obs_arr)
        labels_plot.append("CombiPrecip")
        lengths.append((xlen, ylen))

        # determine the max bound from High resolution only
        combi_vals = obs_arr.flatten()
        combi_vals = combi_vals[np.isfinite(combi_vals) & (combi_vals > 0)]
        combiprecip_max = combi_vals.max() if combi_vals.size > 0 else None

        suffix = f"_lead{lead + 1}"
        if arrays:
            fig_pdf, _ = plot_log_pdfs(
                arrays,
                labels_plot,
                colors=[colors[list(filenames).index(l)] for l in labels_plot],
                ls=[ls[list(filenames).index(l)] for l in labels_plot],
                force_max_x=combiprecip_max
            )
            fig_pdf.savefig(os.path.join(figs_dir, f"logpdf_comparison{suffix}.png"))
            plt.close(fig_pdf)



def main():
    with open(os.path.join(os.path.dirname(__file__), "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    base = dirs["main"]["base"]

    filenames = {
        "S2S": os.path.join(base, dirs["subs"]["test_data"], "det_s2s_nearest_low-pass_1h.h5"),
        "S2S Interpolated": os.path.join(base, dirs["subs"]["test_data"], "det_s2s_nearest_low-pass_1h_interp.h5"),
        "Diffusion Spatial": os.path.join(base, dirs["subs"]["simulations"], "diffusion/light_longer_cli100_ens4.h5"),
        "Diffusion Spatial Interp": os.path.join(base, dirs["subs"]["simulations"], "diffusion/light_longer_cli100_ens4_1h_interp.h5"),
        "Diffusion Spatio-temporal": os.path.join(base, dirs["subs"]["simulations"], "diffusion/light_longer_st_cli100_ens4.h5"),
        "CombiPrecip": os.path.join(base, dirs["subs"]["test_data"], "cpc_condtday1h.h5"),
    }

    labels = list(filenames)
    colors = [CURVE_CMAP(i) for i in range(len(labels))]
    ls = ['-'] * len(labels)

    make_logpdf_plots(filenames, colors=colors, ls=ls)



if __name__ == "__main__":
    main()
    print("Done!")
