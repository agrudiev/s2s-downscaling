import h5py, os, tomllib
import jax, optax
import orbax.checkpoint as ocp
from clu import metric_writers

from swirl_dynamics import templates
from swirl_dynamics.lib import diffusion as dfn_lib
from swirl_dynamics.projects import probabilistic_diffusion as dfn

import configs
from dataset_utils import get_dataset_daytime

def train(
    train_file_path,
    validation_file_path,
    workdir,
    key,
    data_std,
    num_channels,
    downsample_ratio,
    num_blocks,
    num_train_steps,
    train_batch_size,
    eval_batch_size,
    initial_lr,
    peak_lr,
    warmup_steps,
    end_lr,
    ema_decay,
    ckpt_interval,
    max_ckpt_to_keep,
):
    # ************
    # Architecture 
    # ************
    cond_denoiser_model = dfn_lib.PreconditionedDenoiserUNet(
        out_channels=1,
        resize_to_shape=(224, 336),
        num_channels=num_channels,
        downsample_ratio=downsample_ratio,
        num_blocks=num_blocks,
        noise_embed_dim=128,
        cond_resize_method="cubic",
        cond_embed_dim=128,
        padding="SAME",
        use_attention=True,
        use_position_encoding=True,
        num_heads=8,
        sigma_data=data_std,
    )
    
    # **************
    # Training setup
    # **************
    diffusion_scheme = dfn_lib.Diffusion.create_variance_exploding(
        sigma=dfn_lib.tangent_noise_schedule(),
        data_std=data_std,
    )

    with h5py.File(validation_file_path, 'r') as f:
        Nx = f['longitude'][:].shape[0]
        Ny = f['latitude'][:].shape[0]

    cond_model = dfn.DenoisingModel(
        input_shape=(Nx, Ny, 1),
        denoiser=cond_denoiser_model,
        cond_shape = {
            "channel:daytime_cos": (230, 341, 1),
            "channel:daytime_sin": (230, 341, 1),
        },
        noise_sampling=dfn_lib.log_uniform_sampling(
            diffusion_scheme, clip_min=1e-4, uniform_grid=True,
        ),
        noise_weighting=dfn_lib.edm_weighting(data_std=data_std)
    )

    # *****
    # Train
    # *****
    cond_trainer = dfn.DenoisingTrainer(
        model=cond_model,
        rng=jax.random.PRNGKey(888),
        optimizer=optax.adam(
            learning_rate=optax.warmup_cosine_decay_schedule(
                init_value=initial_lr,
                peak_value=peak_lr,
                warmup_steps=warmup_steps,
                decay_steps=num_train_steps,
                end_value=end_lr,
            ),
        ),
        # We keep track of an exponential moving average of the model parameters
        # over training steps. This alleviates the "color-shift" problems known to
        # exist in the diffusion models.
        ema_decay=ema_decay,
    )
    
    templates.run_train(
        train_dataloader=get_dataset_daytime(
            file_path=train_file_path,
            key=key,
            batch_size=train_batch_size,
        ),
        trainer=cond_trainer,
        workdir=workdir,
        total_train_steps=num_train_steps,
        metric_writer=metric_writers.create_default_writer(
            workdir, asynchronous=False
        ),
        metric_aggregation_steps=100,
        eval_dataloader=get_dataset_daytime(
            file_path=validation_file_path,
            key=key,
            batch_size=eval_batch_size,
        ),
        eval_every_steps = 1000,
        num_batches_per_eval = eval_batch_size,
        callbacks=(
            # This callback displays the training progress in a tqdm bar
            templates.TqdmProgressBar(
                total_train_steps=num_train_steps,
                train_monitors=("train_loss",),
            ),
            # This callback saves model checkpoint periodically
            templates.TrainStateCheckpoint(
                base_dir=workdir,
                options=ocp.CheckpointManagerOptions(
                    save_interval_steps=ckpt_interval, max_to_keep=max_ckpt_to_keep
                ),
            ),
        ),
    )


def main():
    # directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    base = dirs["main"]["base"]
    train_data_dir = os.path.join(base, dirs["subs"]["train_data"])
    validation_data_dir = os.path.join(base, dirs["subs"]["validation_data"])
    
    # extra configurations
    config = configs.light_longer_st.get_config(train_data_dir, validation_data_dir)
    train_file_path = config.train_file_path
    validation_file_path = config.validation_file_path
    workdir = config.workdir
    key = config.key
    data_std = config.data_std
    num_channels = config.num_channels
    downsample_ratio = config.downsample_ratio
    num_blocks = config.num_blocks
    num_train_steps = config.num_train_steps
    train_batch_size = config.train_batch_size
    eval_batch_size = config.eval_batch_size
    initial_lr = config.initial_lr
    peak_lr = config.peak_lr
    warmup_steps = config.warmup_steps
    end_lr = config.end_lr
    ema_decay = config.ema_decay
    ckpt_interval = config.ckpt_interval
    max_ckpt_to_keep = config.max_ckpt_to_keep

    # main call    
    train(
        train_file_path,
        validation_file_path,
        workdir,
        key,
        data_std,
        num_channels,
        downsample_ratio,
        num_blocks,
        num_train_steps,
        train_batch_size,
        eval_batch_size,
        initial_lr,
        peak_lr,
        warmup_steps,
        end_lr,
        ema_decay,
        ckpt_interval,
        max_ckpt_to_keep,
    )
    

if __name__ == "__main__":
    main()
