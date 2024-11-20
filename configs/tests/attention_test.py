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
    config.train_molecules = (0, 64)
    config.val_molecules = (64, 96)

    config.model = ml_collections.ConfigDict()
    config.model.model_name = "attention-unet"
    config.model.output_channels = 5
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
    config.model.return_attention_maps = False

    return config
