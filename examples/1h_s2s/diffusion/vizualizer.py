import os
import h5py
import tomllib
import numpy as np

from evaluation.plots import plot_maps_vizu


def main():
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)

    base = dirs["main"]["base"]
    test_data_dir = os.path.join(base, dirs["subs"]["test_data"])
    simulations_dir = os.path.join(base, dirs["subs"]["simulations"])

    s2s_path = os.path.join(test_data_dir, "det_s2s_nearest_low-pass_1h.h5")
    s2s_interp_path = os.path.join(test_data_dir, "det_s2s_nearest_low-pass_1h_interp.h5")
    sim_path = os.path.join(simulations_dir, "diffusion/light_longer_cli100_ens4.h5")
    sim_interp_path = os.path.join(simulations_dir, "diffusion/light_longer_cli100_ens4_1h_interp.h5")
    sim2_path = os.path.join(simulations_dir, "diffusion/light_longer_st_cli100_ens4.h5")
    cpc_data_path = os.path.join(test_data_dir, "cpc_condtday1h.h5")

    
    with h5py.File(s2s_path, "r") as f:
        s2s = f["precip"][0]
    with h5py.File(s2s_interp_path, "r") as f:
        s2s_interp = f["precip"][0]
    with h5py.File(sim_path, "r") as f:
        diffusion_s = f["precip"][0, 0]
    with h5py.File(sim_interp_path, "r") as f:
        diffusion_interp = f["precip"][0]
    with h5py.File(sim2_path, "r") as f:
        diffusion_st = f["precip"][0, 0]
    with h5py.File(cpc_data_path, "r") as f:
        cpc_data = f["precip"][:]

    
    output_path = os.path.join(script_dir, "figs", "comparison_panel.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data_rows = [
        [s2s[2], None, None, None, None, None, s2s[3]],                  # S2S 
        [s2s_interp[t] for t in range(12, 19)],                          # S2S Interp (t=6 à 12)
        [diffusion_s[2], None, None, None, None, None, diffusion_s[3]],  # Spatial Diffusion 
        [diffusion_interp[t] for t in range(12, 19)],
        [diffusion_st[t] for t in range(12, 19)],                        # Spatio-temporal Diffusion
        [cpc_data[t] for t in range(12, 19)]                             # CPC
    ]

    row_titles = [
        "S2S",
        "S2S \nInterpolated",
        "Diffusion \nSpatial",
        "Diffusion \nSpatial \nInterpolated",
        "Diffusion \nSpatio-temporal  ",
        "CPC"
    ]
    col_titles = [
        ["t=12", None, None, None, None, None, "t=18"],                   # S2S
        [f"t={t}" for t in range(12, 19)],                                # S2S Interp
        ["t=12", None, None, None, None, None, "t=18"],                   # Spatial Diffusion
        [f"t={t}" for t in range(12, 19)],
        [f"t={t}" for t in range(12, 19)],                                # Spatio-temporal Diffusion
        [f"t={t}" for t in range(12, 19)]                                 # CPC
    ]

    print("Shapes:")
    for row in data_rows:
        print([arr.shape if arr is not None else None for arr in row])


    plot_maps_vizu(
        data_rows=data_rows,
        row_titles=row_titles,
        col_titles=col_titles,
        extent=[5.8, 10.6, 45.6, 47.9],
        save_path=output_path
    )


if __name__ == "__main__":
    main()
    print("Done!")
