from configs import default

import ml_collections

def get_config() -> ml_collections.ConfigDict:
    """Get the default configuration."""
    config = default.get_config()
    config.debug = True

    config.num_train_steps = 50
    config.num_eval_steps = 10
    config.log_every_steps = 10
    config.eval_every_steps = 25
    config.predict_every_steps = 50
    config.predict_num_batches = 1
    config.predict_num_batches_at_end_of_training = 1
    
    config.batch_size = 4
    config.train_molecules = (0, 2000)
    config.val_molecules = (2000, 3000)

    config.model = ml_collections.ConfigDict()
    config.model.model_name = "unet"
    config.model.output_channels = 5
    config.model.filters = [2, 4, 8]
    config.model.kernel_size = [3, 3, 3]

    return config
