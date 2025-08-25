import os, shutil
import jax, h5py
import numpy as np
import jax.numpy as jnp
import xarray as xr
from math import radians, sin, cos, sqrt, atan2
from typing import Dict


# Utils function in (almost) alfabetical order

def batch_mul(a, b):
    """
    Multiply for each batch.
    """
    return jax.vmap(lambda a, b: a * b)(a, b)


def create_folder(path, overwrite=False):
    """
    Creates a folder at the specified path, deleting the existing folder if overwrite is True.
    """
    if os.path.exists(path) & overwrite:
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    
    
def delog_transform(data, epsilon=1e-4):
    """
    Inverse of log transformation.
    """
    return jnp.exp(data + jnp.log(epsilon)) - epsilon


def denormalize_data(data, mean, std, norm_mean=0.0, norm_std=1.0):
    """
    Inverse of normalization.
    """
    return (data-norm_mean) * std / norm_std + mean


def deprocess_data(data, mean, std, norm_mean=0.0, norm_std=1.0, is_log_transforming=True, clip_zero=False):
    """
    Inverse of process_data.
    """
    data = denormalize_data(data, mean, std, norm_mean, norm_std)
    if is_log_transforming:
        data = delog_transform(data)
    if clip_zero:
        data = jnp.clip(data, 0, None)
    return data


def filter_dry_images(data, threshold=0.1, fraction=0.2, seed=123, return_indices=False):
    """
    Filters out very dry images from the input data based on the mean rain information.
    This is useful to avoid overweighting the model to no rain events.
    """
    # Calculate the mean rain information for each image
    mean_rain = jnp.mean(data, axis=(1, 2))
    
    # Find the indices of very dry images
    dry_indices = jnp.where(mean_rain < threshold)[0]
    
    # Randomly select 1/4 of the very dry images
    num_to_select = int(len(dry_indices) * fraction)
    rng = jax.random.PRNGKey(seed)
    selected_indices = jax.random.choice(rng, dry_indices, shape=(num_to_select,), replace=False)
    
    # Find the indices of images that are not very dry
    non_dry_indices = jnp.where(mean_rain >= threshold)[0]
    
    # Combine the selected dry indices with the non-dry indices
    final_indices = jnp.concatenate([selected_indices, non_dry_indices])
    
    if return_indices:
        return final_indices
    else:
        return data[final_indices]


def find_intersection(wavelengths, era5_psd, cpc_psd, threshold=1):
    """
    Finds the intersection point of two curves (era5_psd and cpc_psd) based on the given wavelengths.
    """
    # Find the differences between the two curves
    y1 = np.log(era5_psd)[1:] # there's a problem with the first value?
    y2 = np.log(cpc_psd)[1:]
    diff = y2 - y1

    # Find the index of the minimum absolute difference (closest to zero)
    idx = np.where(diff < threshold)[0][0]

    # Return the wavelength and y-value at the intersection point
    lambda_star = wavelengths[idx+1 +1]
    psd_star = cpc_psd[idx+1 +1]

    return lambda_star, psd_star


def gaussian_noise(data, sigma):
    """
    Adds Gaussian noise to the input data with the specified standard deviation (sigma).
    """
    noise = np.random.normal(0, sigma, data.shape)
    return data + noise


def get_spatial_lengths(lons, lats):
    """
    Calculates the spatial lengths (x_length and y_length) based on the given longitude and latitude values.
    """
    # Get spatial lengths
    x_length = haversine(lons.min(), lats.min(), lons.max(), lats.min())
    y_length = haversine(lons.min(), lats.min(), lons.min(), lats.max())
    
    return x_length, y_length

# Utils function in alfabetical order

def batch_add(a, b):
    """
    Sum for each batch.
    """
    return jax.vmap(lambda a, b: a + b)(a, b)


def batch_mul(a, b):
    """
    Multiply for each batch.
    """
    return jax.vmap(lambda a, b: a * b)(a, b)


def create_folder(path, overwrite=False):
    """
    Creates a folder at the specified path, deleting the existing folder if overwrite is True.
    """
    if os.path.exists(path) & overwrite:
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def find_intersection(wavelengths, era5_psd, cpc_psd, threshold=1):
    """
    Finds the intersection point of two curves (era5_psd and cpc_psd) based on the given wavelengths.
    """
    # Find the differences between the two curves
    y1 = np.log(era5_psd + 10e-15)[1:] # there's a problem with the first value?
    y2 = np.log(cpc_psd + 10e-15)[1:]
    diff = y2 - y1

    # Find the index of the minimum absolute difference (closest to zero)
    idx = np.where(diff < threshold)[0][0]

    # Return the wavelength and y-value at the intersection point
    lambda_star = wavelengths[idx+1]
    psd_star = cpc_psd[idx+1]

    return lambda_star, psd_star


def get_precip_dims_dict(file_path: str):
    """
    Reads the dimensions of the dataset from the specified .h5 file
    as the attributes with only one dimension.
    """
    # maybe this can be improved in the future to something more general
    with h5py.File(file_path, "r") as f:
        dims_dict = {}
        for key in f.keys():
            if len(f[key].shape) == 1:
                dims_dict[key] = f[key][:]
                
    return dims_dict


def get_cdf(x, bins):
    """
    Calculates the Cumulative Distribution Function (CDF) of the input data based on the given bins.
    """
    pdf = get_pdf(x, bins)
    cdf = np.cumsum(pdf)
    cdf = np.insert(cdf, 0, 0.0)
    return cdf / cdf.max()


def get_pdf(x, bins):
    """
    Calculates the Probability Density Function (PDF) of the input data based on the given bins.
    """
    pdf, _ = np.histogram(x, bins)
    return pdf / pdf.sum()


def get_spatial_lengths(lons, lats):
    """
    Calculates the spatial lengths (x_length and y_length) based on the given longitude and latitude values.
    """
    # Get spatial lengths
    x_length = haversine(lons.min(), lats.min(), lons.max(), lats.min())
    y_length = haversine(lons.min(), lats.min(), lons.min(), lats.max())
    
    return x_length, y_length


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculates the distance between two points on the Earth's surface using the Haversine formula.
    """
    # Convert latitude and longitude from degrees to radians
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    R = 6371  # Radius of the Earth in kilometers
    distance = R * c
    
    return distance


def log_transform(data, epsilon=1e-4):
    """
    Applies a log transformation to the input data using the formula x̃ = log(x + ϵ) − log(ϵ).
    """
    return jnp.log(data + epsilon) - jnp.log(epsilon)


def normalize_data(data, mean=None, std=None, norm_mean=0.0, norm_std=1.0):
    """
    Normalizes the input data to have zero mean and the specified standard deviation (std_data).
    If mean and std are provided, they are used; otherwise, they are inferred from the data.
    """
    if mean is None:
        mean = jnp.mean(data)
    if std is None:
        std = jnp.std(data)
    normalized_data = (data - mean) * norm_std / std + norm_mean
    return normalized_data, mean, std


def process_data(data, mean, std, norm_mean=0.0, norm_std=1.0, is_log_transforming=True):
    """
    Apply log transformation (if True) and normalization to the input data with given std_data.
    """
    if is_log_transforming:
        data = log_transform(data)
    data, mean, std = normalize_data(data, mean, std, norm_mean, norm_std)
    return data, mean, std


def take_nan_imgs_out(data):
    idx_to_keep = []
    for i in range(data.shape[0]):
        if not np.isnan(data[i]).any():
            idx_to_keep.append(i)
    data = data[idx_to_keep]
    return data


def _write_deterministic_dataset(times, dyear, time_daytime, lats, lons, data, filename):
    """
    Creates an xarray dataset from the input data and writes it to a .h5 file at the specified filename.
    """
    # Create dataset
    ds = xr.Dataset(
        {
            "precip": (["time", "latitude", "longitude"], data),
            "dyear": (["time"], dyear),  # Nouvelle variable
            "time_daytime": (["time"], time_daytime)  # Ajout de time_daytime
        },
        coords={
            "time": (["time"], times),
            "latitude": (["latitude"], lats),
            "longitude": (["longitude"], lons),
        },
    )

    # Write dataset
    ds.to_netcdf(filename, engine="h5netcdf")


def _write_ensemble_dataset(times, dyear, time_daytime, lats, lons, data, filename):
    """
    Creates an xarray dataset from the input data and writes it to a .h5 file at the specified filename.
    """
    # Create dataset
    ds = xr.Dataset(
        {
            "precip": (["time", "ensemble", "latitude", "longitude"], data),
            "dyear": (["time"], dyear),  # Nouvelle variable
            "time_daytime": (["time"], time_daytime)  # Ajout de time_daytime
        },
        coords={
            "time": (["time"], times),
            "ensemble": (["ensemble"], np.arange(data.shape[1])),
            "latitude": (["latitude"], lats),
            "longitude": (["longitude"], lons),
        },
    )

    # Write dataset
    ds.to_netcdf(filename, engine="h5netcdf")


    
def write_dataset(times, dyear, time_daytime, lats, lons, data, filename):
    """
    Creates an xarray dataset from the input data and writes it to a .h5 file at the specified filename.
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    if len(data.shape) == 4:
        _write_ensemble_dataset(times, dyear, time_daytime, lats, lons, data, filename)
    elif len(data.shape) == 3:
        _write_deterministic_dataset(times, dyear, time_daytime, lats, lons, data, filename)
    else:
        raise ValueError("Data must have either 3 (deterministic) or 4 (ensemble) dimensions.")  
 
  
def write_precip_to_h5(dims: dict[str, np.ndarray], data: np.ndarray, filename: str):
    """
    Writes the input data to a .h5 file at the specified filename following the specified dimensions.
    """
    dim_keys = list(dims.keys())
    
    if data.shape != tuple(len(dims[dim]) for dim in dim_keys):
        raise ValueError("Shape of data does not match the dimensions in dims.")

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
        
    ds = xr.Dataset(
        {
            "precip": (dim_keys, data)
        },
        coords=dims
    )

    ds.to_netcdf(filename, engine="h5netcdf")
