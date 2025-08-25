import h5py, os, tomllib
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from swirl_dynamics.lib import diffusion as dfn_lib
from swirl_dynamics.lib import solvers as solver_lib
from swirl_dynamics.projects import probabilistic_diffusion as dfn
from utils import write_precip_to_h5

import configs
from dataset_utils import get_dataset_info, get_normalized_test_dataset


def generate(config, file_path, save_path, clip_max, num_samples):
    # Get denoiser model back
    denoiser_model = dfn_lib.PreconditionedDenoiserUNet(
        out_channels=1,
        resize_to_shape=(224, 336),
        num_channels=config.num_channels,
        downsample_ratio=config.downsample_ratio,
        num_blocks=config.num_blocks,
        noise_embed_dim=128,
        padding="SAME",
        use_attention=True,
        use_position_encoding=True,
        num_heads=8,
        sigma_data=config.data_std,
    )
    
    # Restore train state from checkpoint. By default, the move recently saved
    # checkpoint is restored. Alternatively, one can directly use
    # `trainer.train_state` if continuing from the training section above.
    trained_state = dfn.DenoisingModelTrainState.restore_from_orbax_ckpt(
        f"{config.workdir}/checkpoints", step=None
    )
    # Construct the inference function
    denoise_fn = dfn.DenoisingTrainer.inference_fn_from_state_dict(
        trained_state, use_ema=True, denoiser=denoiser_model
    )
    
    # Schemes
    diffusion_scheme = dfn_lib.Diffusion.create_variance_exploding(
        sigma=dfn_lib.tangent_noise_schedule(),
        data_std=config.data_std,
    )
    new_diffusion_scheme = dfn_lib.Diffusion.create_variance_exploding(
        sigma=dfn_lib.tangent_noise_schedule(clip_max=clip_max),
        data_std=config.data_std,
    )
    
    # Get train dataset info
    train_shape, train_mean, train_std = get_dataset_info(
        file_path=config.train_file_path,
        key="precip",
    )

    # Sampler
    sampler = dfn_lib.SdeCustomSampler(
        input_shape=train_shape[1:],
        integrator=solver_lib.EulerMaruyama(),
        tspan=dfn_lib.edm_noise_decay(
            new_diffusion_scheme, rho=7, num_steps=256, end_sigma=1e-3,
        ),
        scheme=new_diffusion_scheme,
        denoise_fn=denoise_fn,
        guidance_transforms=(),
        apply_denoise_at_end=True,
        return_full_paths=False, # Set to `True` if the full sampling paths are needed
    )
    
    # JIT sampler and sample
    generate = jax.jit(sampler.generate, static_argnames=('num_samples',))
    
    # Test dataset
    test_ds = get_normalized_test_dataset(
        file_path=file_path,
        key="precip",
    )
    
    # Add missing dimensions to simulate a 5D array (samples and channel=1)
    if test_ds.ndim == 5:  # (lead_time, time, lat, lon, 1)
        test_ds = test_ds[:, np.newaxis, :, :, :, :]  # → (lead_time, samples, time, lat, lon, channel=1)

    # Now we can extract the dimensions
    num_lead_times, _, num_times, num_lats, num_lons, num_channels = test_ds.shape

    # Check the shape
    print("Final shape of test_ds:", test_ds.shape)  # Expected shape: (lead_time, samples, time, lat, lon, channel=1)

    # Redefine the shape of samples_array to store the generated samples
    samples_array = jnp.zeros((num_lead_times, num_samples, num_times, num_lats, num_lons, num_channels))

    # Check the shape of samples_array
    print("Shape of samples_array:", samples_array.shape)

    # Initialize the PRNG key for random generation in JAX
    rng = jax.random.PRNGKey(0)


    with tqdm(total=num_lead_times * num_times, desc="Generating samples") as pbar:
        for lead_time_idx in range(num_lead_times):
            for time_idx in range(num_times):
                rng, rng_step = jax.random.split(rng)

                
                input_sample = test_ds[lead_time_idx, 0, time_idx]

                samples = generate(
                    init_sample=input_sample,
                    rng=rng_step,
                    num_samples=num_samples,
                ) * train_std + train_mean

                samples_array = samples_array.at[
                    lead_time_idx, :, time_idx
                ].set(samples)

                pbar.update(1)

    # Clip
    samples = jnp.clip(samples_array[..., 0], a_min=0)

    with h5py.File(file_path, "r") as f:
        lead_times = f["lead_time"][:]
        times = f["time"][:]
        lats = f["latitude"][:]
        lons = f["longitude"][:]

    dims_dict = {
        "lead_time": lead_times,
        "ensemble": np.arange(num_samples), #
        "time": times,
        "latitude": lats,
        "longitude": lons,
    }

    write_precip_to_h5(dims_dict, samples, save_path)
    

def main():
    # directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    base = dirs["main"]["base"]
    train_data_dir = os.path.join(base, dirs["subs"]["train_data"])
    validation_data_dir = os.path.join(base, dirs["subs"]["validation_data"])
    test_data_dir = os.path.join(base, dirs["subs"]["test_data"])
    simulations_dir = os.path.join(base, dirs["subs"]["simulations"])
    
    # extra configurations
    model_config = configs.light_longer.get_config(train_data_dir, validation_data_dir)
    prior_file_path = os.path.join(test_data_dir, "det_s2s_nearest_low-pass.h5")
    clip_max = 100
    num_samples = 4
    save_file_path = os.path.join(
        simulations_dir,
        "diffusion",
        f"{model_config.experiment_name}_cli{clip_max}_ens{num_samples}.h5",
    )
    
    # main call
    generate(model_config, prior_file_path, save_file_path, clip_max, num_samples)


if __name__ == "__main__":
    main()
    print("Done generating")
