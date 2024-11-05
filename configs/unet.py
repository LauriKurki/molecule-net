import ml_collections
from configs import default

def get_model_config() -> ml_collections.ConfigDict:
    """Get hyperparameters for the UNet model."""

    config = ml_collections.ConfigDict()
    config.model_name = "UNet"
    config.output_channels = 5
    config.filters = [16, 32, 64]
    config.kernel_size = [3, 3, 3]

    return config


def get_config() -> ml_collections.ConfigDict:
    """Get the default configuration."""
    config = default.get_config()
    config.model = get_model_config()

    return config
