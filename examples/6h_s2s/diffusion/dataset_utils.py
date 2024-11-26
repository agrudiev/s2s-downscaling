import jax.numpy as jnp
import tensorflow as tf
import xarray as xr
from swirl_dynamics.data.hdf5_utils import read_single_array

# TODO: create functionality to apply log during training (and take it out in test)

def get_dataset(file_path: str, key: str, batch_size: int, apply_log: bool=False):
    # Read the dataset from the .hdf5 file.
    images = read_single_array(file_path, key)

    # Normalize the images
    mu = jnp.mean(images)
    sigma = jnp.std(images)
    images = (images - mu) / sigma

    # Expand dims
    images = jnp.expand_dims(images, axis=-1)

    # Create a TensorFlow dataset from the images.
    ds = tf.data.Dataset.from_tensor_slices({"x": images})

    # Repeat, batch, and prefetch the dataset.
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.as_numpy_iterator()

    return ds


def get_dataset_info(file_path: str, key: str, apply_log: bool=False):
    # Read the dataset from the .hdf5 file.
    images = read_single_array(file_path, key)
    
    # Normalize the images
    mu = jnp.mean(images)
    sigma = jnp.std(images)
    images = (images - mu) / sigma

    return images.shape, mu, sigma


def get_normalized_test_dataset(file_path: str, key: str, apply_log: bool=False):
    images = read_single_array(file_path, key)

    mu = jnp.mean(images)
    sigma = jnp.std(images)
    images = (images - mu) / sigma

    ds = jnp.expand_dims(images, axis=-1)

    return ds
