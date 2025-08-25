import os
import h5py
import tomllib
import numpy as np
from pysteps.verification import sal
from evaluation.metrics import root_mean_squared_error


def load_dirs():
    with open(os.path.join(os.path.dirname(__file__), "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    base = dirs["main"]["base"]
    return {
        "S2S": os.path.join(base, dirs["subs"]["test_data"], "det_s2s_nearest_low-pass_1h.h5"),
        "S2S_interp": os.path.join(base, dirs["subs"]["test_data"], "det_s2s_nearest_low-pass_1h_interp.h5"),
        "Diffusion": os.path.join(base, dirs["subs"]["simulations"], "diffusion/light_longer_cli100_ens4.h5"),
        "DiffusionInterp": os.path.join(base, dirs["subs"]["simulations"], "diffusion/light_longer_cli100_ens4_1h_interp.h5"),
        "DiffusionST": os.path.join(base, dirs["subs"]["simulations"], "diffusion/light_longer_st_cli100_ens4.h5"),
        "CombiPrecip": os.path.join(base, dirs["subs"]["test_data"], "cpc_condtday1h.h5"),
    }

# Extract the observation image at a given time index
def get_obs_field(obs_path, time_index):
    with h5py.File(obs_path, 'r') as f:
        obs = f["precip"]
        return obs[time_index]

# Extract 3 precipitation fields from the model (at the same time t)
def get_model_fields(model_path):
    with h5py.File(model_path, 'r') as f:
        p = f["precip"]
        shape = p.shape

        if len(shape) == 5:       # (leadtime, ensemble, time, x, y)
            return [p[lt, 0, 0] for lt in range(3)]
        elif len(shape) == 4:     # (leadtime, time, x, y)
            return [p[lt, 0] for lt in range(3)]
        elif len(shape) == 3:     # (time, x, y)
            return [p[0] for _ in range(3)]
        else:
            raise ValueError(f"Unsupported precip shape: {shape}")

def main():
    paths = load_dirs()
    obs_path = paths["CombiPrecip"]
    obs_time_index = 39  # Temporal index of the reference image

    print("\n=== SAL & RMSE Scores à t = 39 ===\n")

    for model_name, model_path in paths.items():
        if model_name == "CombiPrecip":
            continue  # We do not compare the observation to itself

        try:
            preds = get_model_fields(model_path)
            obs = get_obs_field(obs_path, obs_time_index)

            print(f"{model_name}:")
            for i, pred in enumerate(preds):
                if np.all(pred == 0) or np.all(obs == 0):
                    print(f"  Champ {i}: Image vide (tout 0)")
                    continue

                s, a, l = sal(pred, obs, thr_quantile=0.95, thr_factor=0.067)
                rmse = root_mean_squared_error(pred, obs)

                print(f"  Champ {i}: S={s:.3f}, A={a:.3f}, L={l:.3f}, RMSE={rmse:.3f}")
        except Exception as e:
            print(f"{model_name}: Erreur -> {e}")

if __name__ == "__main__":
    main()
