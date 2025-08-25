import os
import tomllib
import h5py
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from swirl_dynamics.lib import diffusion as dfn_lib
from swirl_dynamics.lib import solvers as solver_lib
from swirl_dynamics.projects import probabilistic_diffusion as dfn

from utils import write_precip_to_h5
from dataset_utils import get_dataset_info, get_normalized_test_dataset
import configs


def generate(config, file_path, save_path, clip_max, num_samples):
    # Model
    cond_denoiser_model = dfn_lib.PreconditionedDenoiserUNet(
        out_channels=1,
        resize_to_shape=(224, 336),
        num_channels=config.num_channels,
        downsample_ratio=config.downsample_ratio,
        num_blocks=config.num_blocks,
        noise_embed_dim=128,
        cond_resize_method="cubic",
        cond_embed_dim=128,
        padding="SAME",
        use_attention=True,
        use_position_encoding=True,
        num_heads=8,
        sigma_data=config.data_std,
    )

    trained_state = dfn.DenoisingModelTrainState.restore_from_orbax_ckpt(
        f"{config.workdir}/checkpoints"
    )
    denoise_fn = dfn.DenoisingTrainer.inference_fn_from_state_dict(
        trained_state, use_ema=True, denoiser=cond_denoiser_model
    )

    diffusion_scheme = dfn_lib.Diffusion.create_variance_exploding(
        sigma=dfn_lib.tangent_noise_schedule(clip_max=clip_max),
        data_std=config.data_std,
    )

    train_shape, train_mean, train_std = get_dataset_info(config.train_file_path, key="precip")

    cond_sampler = dfn_lib.SdeCustomSampler(
        input_shape=train_shape[1:],
        integrator=solver_lib.EulerMaruyama(),
        tspan=dfn_lib.edm_noise_decay(diffusion_scheme, rho=7, num_steps=256, end_sigma=1e-3),
        scheme=diffusion_scheme,
        denoise_fn=denoise_fn,
        apply_denoise_at_end=True,
    )

    generate_fn = jax.jit(cond_sampler.generate, static_argnames=("num_samples",))

    # Load preprocessed dataset
    test_ds = get_normalized_test_dataset(file_path=file_path, key="precip")
    if test_ds.ndim == 5:
        test_ds = test_ds[:, np.newaxis]  # (lead_time, 1, time, H, W, 1)

    num_lead_times, _, num_times, H, W, _ = test_ds.shape

    with h5py.File(file_path, "r") as f:
        lead_times = f["lead_time"][:]
        times = f["time"][:]
        daytime = f["daytime"][:]  # (num_times, 2)
        lats = f["latitude"][:]
        lons = f["longitude"][:]

    # Allocate output array
    samples_array = jnp.zeros((num_lead_times, num_samples, num_times, H, W, 1))
    rng = jax.random.PRNGKey(0)

    for lead_time_idx in range(num_lead_times):
        print(f"\nLead time {lead_time_idx + 1}/{num_lead_times}")
        with tqdm(total=num_times, desc=f"Generating samples", dynamic_ncols=True) as pbar:
            for step_idx in range(num_times):
                input_sample = test_ds[lead_time_idx, 0, step_idx]  # (H, W, 1)
                cos_val, sin_val = daytime[step_idx]

                rng, rng_step = jax.random.split(rng)
                samples = generate_fn(
                    init_sample=input_sample,
                    rng=rng_step,
                    num_samples=num_samples,
                    cond={
                        "channel:daytime_cos": jnp.full((H, W, 1), cos_val),
                        "channel:daytime_sin": jnp.full((H, W, 1), sin_val),
                    },
                ) * train_std + train_mean

                samples_array = samples_array.at[lead_time_idx, :, step_idx].set(samples)
                pbar.update(1)

    # Finalize
    samples = jnp.clip(samples_array[..., 0], a_min=0)  # (lead_time, ensemble, time, lat, lon)

    dims_dict = {
        "lead_time": lead_times,
        "ensemble": np.arange(num_samples),
        "time": times,
        "latitude": lats,
        "longitude": lons,
    }

    expected_shape = tuple(len(dims_dict[k]) for k in ["lead_time", "ensemble", "time", "latitude", "longitude"])
    if samples.shape != expected_shape:
        raise ValueError(f"Mismatch in output shape {samples.shape} vs {expected_shape}")

    write_precip_to_h5(dims_dict, samples, save_path)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)

    base = dirs["main"]["base"]
    train_data_dir = os.path.join(base, dirs["subs"]["train_data"])
    validation_data_dir = os.path.join(base, dirs["subs"]["validation_data"])
    test_data_dir = os.path.join(base, dirs["subs"]["test_data"])
    simulations_dir = os.path.join(base, dirs["subs"]["simulations"])

    model_config = configs.light_longer_st.get_config(train_data_dir, validation_data_dir)
    prior_file_path = os.path.join(test_data_dir, "det_s2s_nearest_low-pass_1h_interp.h5")  
    clip_max = 100
    num_samples = 4

    save_file_path = os.path.join(
        simulations_dir,
        "diffusion",
        f"{model_config.experiment_name}_cli{clip_max}_ens{num_samples}.h5",
    )

    generate(model_config, prior_file_path, save_file_path, clip_max, num_samples)


if __name__ == "__main__":
    main()
    print("Done generating")
