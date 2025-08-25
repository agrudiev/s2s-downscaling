import os
import h5py
import numpy as np
from tqdm import tqdm

def preprocess(file_path, save_path, steps_per_pair=5):
    # Load raw data
    with h5py.File(file_path, "r") as f:
        precip = f["precip"][:]   # (lead_time, time, lat, lon)
        time = f["time"][:]       # shape (num_times,)
        daytime = f["daytime"][:]
        lat = f["latitude"][:]
        lon = f["longitude"][:]
        lead_times = f["lead_time"][:]

    if precip.ndim == 4:
        precip = precip[:, np.newaxis]  # (lead_time, 1, time, lat, lon)

    num_lead_times, _, num_times, H, W = precip.shape

    # Split data into continuous blocks (where large time gaps exist)
    split_indices = np.where(np.diff(time) > 1000)[0] + 1
    blocks = np.split(np.arange(num_times), split_indices)

    all_precip = [[] for _ in range(num_lead_times)]
    all_time = []
    all_daytime = []
    original_images = {}  # {timestamp: image} for verification

    for block in blocks:
        block_time = time[block]
        block_daytime = daytime[block]
        t_regular = np.arange(block_time[0], block_time[-1] + 1)

        for lead_idx in range(num_lead_times):
            block_precip = precip[lead_idx, 0, block]  # (num_images, H, W)
            interp_precip = np.empty((len(t_regular), H, W))

            # Interpolate pixel by pixel
            for y in range(H):
                for x in range(W):
                    series = block_precip[:, y, x]
                    interp_precip[:, y, x] = np.interp(t_regular, block_time, series)

            # Restore original images at their positions
            original_indices = block_time - block_time[0]
            for i, idx in enumerate(original_indices):
                interp_precip[idx] = block_precip[i]
                global_time = t_regular[idx]
                if lead_idx == 0:
                    original_images[global_time] = block_precip[i]

            all_precip[lead_idx].append(interp_precip)

        # Interpolate daytime values
        block_daytime = np.asarray(block_daytime)
        interp_daytime = np.array([
            np.interp(t_regular, block_time, block_daytime[:, 0]),
            np.interp(t_regular, block_time, block_daytime[:, 1])
        ]).T

        # Restore original daytime values
        for i, idx in enumerate(block_time - block_time[0]):
            interp_daytime[idx] = block_daytime[i]

        all_time.extend(t_regular)
        all_daytime.extend(interp_daytime)

    # Assemble final arrays
    out_precip = np.stack([np.concatenate(p, axis=0) for p in all_precip], axis=0)  # (lead_time, time, H, W)
    out_time = np.array(all_time)
    out_daytime = np.array(all_daytime)

    # Strict check for restored original images
    print("\n--- Strict check of original images ---")
    for t_orig, orig_img in original_images.items():
        indices = np.where(out_time == t_orig)[0]
        if len(indices) == 0:
            print(f"[ERROR] Original timestamp {t_orig} is missing from output.")
            continue
        idx = indices[0]
        interp_img = out_precip[0, idx]  # lead_time=0
        if not np.array_equal(orig_img, interp_img):
            print(f"[ NOT IDENTICAL ] at t = {t_orig}")
        else:
            print(f"[ IDENTICAL ] at t = {t_orig}")

    # Save processed data
    with h5py.File(save_path, "w") as f:
        f.create_dataset("precip", data=out_precip)
        f.create_dataset("time", data=out_time)
        f.create_dataset("daytime", data=out_daytime)
        f.create_dataset("latitude", data=lat)
        f.create_dataset("longitude", data=lon)
        f.create_dataset("lead_time", data=lead_times)
        f.create_dataset("ensemble", data=np.arange(num_lead_times))

if __name__ == "__main__":
    file_path = ".../diffusion/light_longer_cli100_ens4_1h.h5"
    save_path = ".../diffusion/light_longer_cli100_ens4_1h_interp.h5"
    preprocess(file_path, save_path, steps_per_pair=5)
    print("\nDone!")
