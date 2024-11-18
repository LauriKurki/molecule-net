from configs import default
from configs.equivariant import unet

import ml_collections

def get_config() -> ml_collections.ConfigDict:
    """Get the default configuration."""
    config = default.get_config()
    config.debug = True
    config.num_workers = 8

    config.num_train_steps = 100
    config.num_eval_steps = 10
    config.log_every_steps = 10
    config.eval_every_steps = 25
    config.predict_every_steps = 50
    config.predict_num_batches = 1
    config.predict_num_batches_at_end_of_training = 1
    
    config.batch_size = 8
    config.batch_order = "torch"

    config.train_molecules = (0, 50000)
    config.val_molecules = (50000, 60000)

    config.model = unet.get_model_config()
    config.rotations = 4
    config.model.encoder_channels = [8, 16, 32]
    config.model.decoder_channels = [32, 16, 8]

    return config
