import jax.numpy as jnp
import tensorflow as tf
import xarray as xr
from swirl_dynamics.data.hdf5_utils import read_single_array

def get_dataset(file_path: str, key: str, batch_size: int):
    # Read the dataset from the .hdf5 file.
    images = read_single_array(file_path, key)

    # Normalize the images
    mu = jnp.mean(images)
    sigma = jnp.std(images)
    images = (images - mu) / sigma

    # Expand dims
    images = jnp.expand_dims(images, axis=-1)
    
    # Reduce set size
    # MAX_DATASET_SIZE = 5000
    # images = images[:MAX_DATASET_SIZE]

    # Create a TensorFlow dataset from the images.
    ds = tf.data.Dataset.from_tensor_slices({"x": images})

    # Repeat, batch, and prefetch the dataset.
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.as_numpy_iterator()

    return ds


def get_dataset_info(file_path: str, key: str):
    # Read the dataset from the .hdf5 file.
    images = read_single_array(file_path, key)
    
    # # Reduce set size
    # MAX_DATASET_SIZE = 5000
    # images = images[:MAX_DATASET_SIZE]

    # Normalize the images
    mu = jnp.mean(images)
    sigma = jnp.std(images)
    images = (images - mu) / sigma

    return images.shape, mu, sigma


def get_test_dataset_info(file_path: str, key: str):
    # Read the dataset from the .hdf5 file.
    images = read_single_array(file_path, key)
    lons = read_single_array(file_path, "longitude")
    lats = read_single_array(file_path, "latitude")
    times = xr.open_dataset(file_path).time.values

    # Normalize the images
    mu = jnp.mean(images)
    sigma = jnp.std(images)
    images = (images - mu) / sigma

    # Expand dims
    ds = jnp.expand_dims(images, axis=-1)

    return ds, lons, lats, times