import numpy as np
import xarray as xr
import pandas as pd

def time_to_circle(hour):
    """
    Converts an hour to circular coordinates [cos(2πh), sin(2πh)].
    """
    return np.array([np.cos(2 * np.pi * hour / 24), np.sin(2 * np.pi * hour / 24)], dtype=np.float32)

def add_daytime_to_file(input_file, output_file):
    # Load the original dataset
    ds = xr.open_dataset(input_file, engine="h5netcdf")

    # Extract hours from time (assumes time is in hours since reference)
    times = ds["time"].values
    hours = pd.to_datetime(times).hour  # hour of day [0, 23)

    # Convert to circular (cos, sin)
    daytime_circular = np.array([time_to_circle(h) for h in hours])

    # Arrondir les valeurs à 6 décimales pour obtenir des valeurs propres comme 1.0, -1.0, ou 0.0
    daytime_circular = np.round(daytime_circular, decimals=6)

    # Add 'coord' dimension (cos/sin) and create new variable
    ds["daytime"] = xr.DataArray(
        daytime_circular,
        dims=("time", "coord"),
        coords={"time": ds["time"], "coord": ["cos", "sin"]},
        name="daytime"
    )

    # Save to new file
    ds.to_netcdf(output_file, engine="h5netcdf")
    print(f"det_s2s_1h created")

if __name__ == "__main__":
    input_path = ".../diffusion/light_longer_cli100_ens4.h5"
    output_path = ".../diffusion/light_longer_cli100_ens4_1h.h5"
    add_daytime_to_file(input_path, output_path)
