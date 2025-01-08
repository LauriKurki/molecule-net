import ml_collections

def get_config() -> ml_collections.ConfigDict:
    """Get the default configuration."""
    config = ml_collections.ConfigDict()
    config.debug = False
    config.root_dir = None
    config.dataset = "afms_rebias"

    # Dataset
    config.rng_seed = 0
    config.noise_std = 0.1
    config.sigma = 0.2
    config.z_cutoff = 2.0
    config.gaussian_factor = 1.0
    config.interpolate_input_z = 20
    config.cutout_probs = [0.5, 0.3, 0.1, 0.05, 0.05]
    config.max_shift_per_slice = 0.02
    config.max_atoms = 54
    config.train_molecules = (0, 230000)
    config.val_molecules = (230000, 264000)
    config.shuffle_datasets = True
    config.batch_size = 32

    # Loss
    config.loss_fn = "mse"

    # Training
    config.num_train_steps = 1_000_000
    config.num_eval_steps = 1000
    config.log_every_steps = 100
    config.eval_every_steps = 10_000

    config.predict_every_steps = 10_000
    config.predict_num_batches = 2
    config.predict_num_batches_at_end_of_training = 10

    # Optimizer.
    config.optimizer = "adamw"
    config.momentum = None
    config.learning_rate = 3e-4
    config.learning_rate_schedule = "constant"
    config.learning_rate_schedule_kwargs = ml_collections.ConfigDict()
    config.learning_rate_schedule_kwargs.init_value = config.get_ref("learning_rate")
    config.learning_rate_schedule_kwargs.peak_value = 2 * config.get_ref(
        "learning_rate"
    )
    config.learning_rate_schedule_kwargs.warmup_steps = 2000
    config.learning_rate_schedule_kwargs.decay_steps = 50000

    # Predictions
    config.peak_threshold = 0.5

    return config
