import os
import tomllib
import h5py
import numpy as np
from evaluation.metrics import *

def get_positive_obs_values(obs_path):
    """Charge toutes les valeurs positives (observation globale)"""
    with h5py.File(obs_path, 'r') as f:
        obs = f["precip"][()]
    return obs[np.isfinite(obs) & (obs > 0)]


def get_all_positive_model_values(model_path, lead_index):
    """Retourne toutes les valeurs positives du modèle pour un lead donné"""
    with h5py.File(model_path, 'r') as f:
        p = f["precip"]
        if p.ndim == 5:  
            data = p[lead_index, 0]  
        elif p.ndim == 4:  
            data = p[lead_index]  
        elif p.ndim == 3: 
            data = p[:]  
        else:
            raise ValueError(f"Shape inattendue: {p.shape}")

    # flattens all maps into 1D
    flat = data[np.isfinite(data) & (data > 0)]
    return flat


def compare_distributions(obs_vals, sim_vals, model_label):
    try:
        if obs_vals.size == 0 or sim_vals.size == 0:
            raise ValueError("Pas de valeurs positives.")
        js = logpdf_distance_logscale(obs_vals, sim_vals, metric="js")
        hell = logpdf_distance_logscale(obs_vals, sim_vals, metric="hellinger")
        print(f"{model_label:35} | JS: {js:.3f} | Hellinger: {hell:.3f}")
    except Exception as e:
        print(f"[WARN] {model_label}: {e}")


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

    print("[INFO] Chargement des observations...")
    obs_vals = get_positive_obs_values(filenames["CombiPrecip"])
    print(f"[INFO] Nombre de pixels positifs observés : {len(obs_vals):,}")

    for lead in range(3):
        print(f"\n=== Lead Time {lead + 1} ===")
        for model_name in ["S2S", "S2S Interpolated", "Diffusion Spatial", "Diffusion Spatial Interp", "Diffusion Spatio-temporal"]:
            sim_vals = get_all_positive_model_values(filenames[model_name], lead)
            label = f"{model_name} (Lead {lead + 1})"
            compare_distributions(obs_vals, sim_vals, label)


if __name__ == "__main__":
    main()
