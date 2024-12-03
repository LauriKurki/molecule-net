import ml_collections
from configs import default

def get_model_config() -> ml_collections.ConfigDict:
    """Get hyperparameters for the UNet model."""

    config = ml_collections.ConfigDict()
    config.model_name = "Attention-UNet"
    config.dtype = "bfloat16"
    config.output_channels = 5
    config.encoder_channels = [16, 32, 64]
    config.decoder_channels = [64, 32, 16]
    config.attention_channels = [32, 32, 32]
    config.encoder_kernel_size = [
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
    ]
    config.decoder_kernel_size = [
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
    ]
    config.conv_activation = "relu"
    config.attention_activation = "sigmoid"
    config.output_activation = None

    config.return_attention_maps = False

    return config


def get_config() -> ml_collections.ConfigDict:
    """Get the default configuration."""
    config = default.get_config()
    config.model = get_model_config()

    return config
