import jax
import jax.numpy as jnp
import flax.linen as nn

import ml_collections

from molnet.models import UNet

def create_model(
    config: ml_collections.ConfigDict
) -> nn.Module:
    """Create a model based on the configuration."""
    model = UNet(
        output_channels=config.output_channels,
        filters=config.filters,
        kernel_size=config.kernel_size
    )
    
    return model