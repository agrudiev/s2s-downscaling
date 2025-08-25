import os
import numpy as np
import h5py
from PIL import Image
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# === Parameters ===
file_path = ".../data/1h_s2s/test_data/det_s2s.h5"
output_dir = ".../data/1h_s2s/simulations/precip_grids_3week"
gif_output_path = os.path.join(output_dir, "precip_3week_2018.gif")

# === Custom colors ===
CUSTOM_PRECIP_COLORS = [
    '#FFFFFF',  # 0 mm/h or masked
    '#FFFFCC',  # Light Yellow
    '#C7E9B4',  # Light Green
    '#7FCDBB',  # Moderate Blue-green
    '#41B6C4',  # Moderate Blue
    '#1D91C0',  # Blue
    '#225EA8',  # Darker Blue
    '#253494',  # Dark Blue
    '#54278F',  # Purple
    '#7A0177',  # Dark
    '#C51B8A'   # Pink
]
cmap = ListedColormap(CUSTOM_PRECIP_COLORS)

# New scale: 0 → 5 mm/h
levels = np.linspace(0, 5, num=len(CUSTOM_PRECIP_COLORS) + 1)
norm = BoundaryNorm(levels, ncolors=cmap.N)

# === Load the file ===
with h5py.File(file_path, 'r') as f:
    lats = f['latitude'][:]
    lons = f['longitude'][:]
    time = f['time'][:]
    lead_times = f['lead_time'][:]
    precip = f['precip'][:]  # (lead_time, time, lat, lon)
    ref_time_str = f['time'].attrs['units'].split('since')[1].strip()
    ref_datetime = datetime.strptime(ref_time_str, "%Y-%m-%d %H:%M:%S")
    all_datetimes = [ref_datetime + timedelta(hours=int(h)) for h in time]
    lead_idx = list(lead_times).index(b'3-week')

    # === Filter data for 2018 ===
    filtered_dates = [date for date in all_datetimes if date.year == 2018]
    filtered_indices = [i for i, date in enumerate(all_datetimes) if date.year == 2018]

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # === Generate plots for each 2018 date ===
    image_files = []
    for idx, date in zip(filtered_indices, filtered_dates):
        precip_slice = precip[lead_idx, idx, :, :]
        masked_precip = np.ma.masked_where(precip_slice <= 0, precip_slice)

        # === Plot without title ===
        fig, ax = plt.subplots(figsize=(10, 5))
        cmap.set_bad(color='white', alpha=0.0)

        mesh = ax.pcolormesh(lon_grid, lat_grid, masked_precip, cmap=cmap, norm=norm, shading='auto')
        cbar = plt.colorbar(mesh, ax=ax, ticks=levels, spacing='uniform')
        cbar.set_label('Precipitation (mm/h)')

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect('equal')

        plt.tight_layout()
        filename = f"precip_3week_{date.strftime('%Y%m%d_%H%M')}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, transparent=True)
        plt.close()
        print(f"Saved: {filename}")
        image_files.append(filename)

# === Load already generated .png images ===
image_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]

# Sort images by chronological order
image_files.sort()

# === Load the images ===
images = [Image.open(os.path.join(output_dir, f)) for f in image_files]

# === Create the animated GIF ===
images[0].save(
    gif_output_path,
    format='GIF',
    save_all=True,
    append_images=images[1:],  # Add remaining images
    duration=500,   # Duration in ms between frames
    loop=0          # GIF loops indefinitely
)

print(f"Done! {gif_output_path}")
