import jax, os
import jax.numpy as jnp
from tqdm import tqdm

from swirl_dynamics.lib import diffusion as dfn_lib
from swirl_dynamics.lib import solvers as solver_lib
from swirl_dynamics.projects import probabilistic_diffusion as dfn
from utils import write_dataset

import configs
from dataset_utils import get_dataset_info, get_test_dataset_info


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
    test_ds, lons, lats, times = get_test_dataset_info(
        file_path=file_path,
        key="precip",
    )
    
    # Forecast samples
    samples_list = []

    # Iterate over the test dataset and generate samples
    rng = jax.random.PRNGKey(0)
    for i in tqdm(range(test_ds.shape[0])):
        rng, rng_step = jax.random.split(rng)
        # Generate samples from the test dataset
        samples = generate(
            init_sample=test_ds[i],
            rng=rng_step,
            num_samples=num_samples,
        ) * train_std + train_mean
        
        # Append samples to the list
        samples_list.append(samples)
    
    # Convert list to a single numpy array
    forecasts = jnp.array(samples_list)[:,:,:,:,0]
    
    # Clip zeros
    forecasts = jnp.clip(forecasts, min=0, max=None)

    # Save all samples in a single HDF5 file
    write_dataset(
        times, lats, lons, forecasts, save_path,
    )
    

def main():
    model_config = configs.heavy.get_config()
    generation_config = configs.generation.get_config()
    
    prior_file_path = generation_config.prior_file_path
    clip_max = generation_config.clip_max
    num_samples = generation_config.num_samples
    save_file_path = os.path.join(
        generation_config.save_dir, 
        f"{model_config.experiment_name}_cli{clip_max}_ens{num_samples}.h5"
    )
    
    generate(model_config, prior_file_path, save_file_path, clip_max, num_samples)


if __name__ == "__main__":
    main()
