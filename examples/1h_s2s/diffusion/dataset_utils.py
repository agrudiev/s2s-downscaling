import jax.numpy as jnp
import tensorflow as tf
import numpy as np
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


# Function to encode daytime into an embedding vector
def preprocess_daytime(daytime):
    daytime = tf.cast(daytime, tf.float32)
    # MLP to encode daytime in a denser space
    daytime_embedding = tf.keras.layers.Dense(128, activation="relu")(daytime)
    return daytime_embedding


def get_dataset_daytime(file_path: str, key: str, batch_size: int):
    images = read_single_array(file_path, key)             # (T, H, W)
    daytime = read_single_array(file_path, "daytime")      # (T, 2), i.e. (cos, sin)

    H, W = images.shape[1], images.shape[2]

    mu = jnp.mean(images)
    sigma = jnp.std(images)
    images = (images - mu) / sigma
    images = jnp.expand_dims(images, axis=-1)              # (T, H, W, 1)

    ds = tf.data.Dataset.from_tensor_slices((images, daytime))

    def format_sample(x, cond):
        # cond[..., 0] = cos, cond[..., 1] = sin
        cos_val = tf.reshape(cond[0], (1, 1, 1))
        sin_val = tf.reshape(cond[1], (1, 1, 1))

        cond_cos = tf.ones((H, W, 1), dtype=tf.float32) * cos_val
        cond_sin = tf.ones((H, W, 1), dtype=tf.float32) * sin_val

        return {
            "x": tf.cast(x, tf.float32),
            "cond": {
                "channel:daytime_cos": cond_cos,
                "channel:daytime_sin": cond_sin,
            }
        }

    ds = ds.map(format_sample, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.as_numpy_iterator()

    return ds


def get_normalized_test_dataset(file_path: str, key: str, apply_log: bool=False):   # from 6_s2s
    images = read_single_array(file_path, key)
    
    # Normalize the images
    mu = jnp.mean(images)
    sigma = jnp.std(images)
    images = (images - mu) / sigma
    
    # Expand dims
    ds = jnp.expand_dims(images, axis=-1)

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

def interpolate_images_in_time(images, times, steps_per_pair=5):
    """
    Interpolate `steps_per_pair` images between each pair of consecutive images
    of a block, and inserts the original images with the correct clues.
    """
    images, times = np.asarray(images), np.asarray(times)

    interpolated_images = []
    interpolated_times = []

    for i in range(len(images) - 1):
        img0, img1 = images[i], images[i + 1]
        t0, t1 = times[i], times[i + 1]

        # interpolation
        for step in range(1, steps_per_pair + 1):
            alpha = step / (steps_per_pair + 1)
            interp_img = (1 - alpha) * img0 + alpha * img1
            interp_time = t0 + alpha * (t1 - t0)

            interpolated_images.append(interp_img)
            interpolated_times.append(interp_time)

    full_images = []
    full_times = []

    for i in range(len(images)):
        full_images.append(images[i])
        full_times.append(times[i])
        if i < len(images) - 1:
            
            idx = i * steps_per_pair
            full_images.extend(interpolated_images[idx:idx + steps_per_pair])
            full_times.extend(interpolated_times[idx:idx + steps_per_pair])

    return np.stack(full_images), np.array(full_times)

def get_interpolated_daytime(daytime_vectors, steps_per_pair=5):
    """
    Interpolate the vectors (cos, sin) between each consecutive pair.
    """
    angles = np.arctan2(daytime_vectors[:, 1], daytime_vectors[:, 0])
    output = [daytime_vectors[0]]

    for i in range(len(angles) - 1):
        a0, a1 = angles[i], angles[i + 1]

        if a1 - a0 > np.pi:
            a1 -= 2 * np.pi
        elif a0 - a1 > np.pi:
            a1 += 2 * np.pi

        for step in range(1, steps_per_pair + 1):
            alpha = step / (steps_per_pair + 1)
            interp_angle = (1 - alpha) * a0 + alpha * a1
            output.append([np.cos(interp_angle), np.sin(interp_angle)])

        output.append(daytime_vectors[i + 1])

    return np.array(output)