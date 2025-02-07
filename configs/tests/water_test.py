from configs import default
from configs import attention_unet

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
    
    config.batch_size = 16
    config.train_molecules = (0, 10000)
    config.val_molecules = (10000, 12000)

    config.dataset = "water-bilayer-tf"
    config.interpolate_input_z = None
    config.target_z_cutoff = 1.0
    config.gaussian_factor = 5.0

    config.model = attention_unet.get_model_config()
    config.model.model_name = "attention-unet"
    config.model.dtype = "bfloat16"
    config.model.output_channels = 2
    config.model.encoder_channels = [2, 4, 8]
    config.model.decoder_channels = [8, 4, 2]
    config.model.attention_channels = [8, 8, 8]
    config.model.encoder_kernel_size = [
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
    ]
    config.model.decoder_kernel_size = [
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
    ]
    config.model.conv_activation = "relu"
    config.model.attention_activation = "sigmoid"
    config.model.output_activation = None

    config.model.return_attention_maps = False

    return config
