import jax
import jax.numpy as jnp
import flax.linen as nn

class ResBlock(nn.Module):
    """Residual block with two convolutional layers and a skip connection."""

    filters: int
    kernel_size: int = 3

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool
    ):
        residual = x
        x = nn.Conv(features=self.filters, kernel_size=self.kernel_size)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.filters, kernel_size=self.kernel_size)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)

        # Projection 
        if x.shape != residual.shape:
            residual = nn.Conv(features=self.filters, kernel_size=1)(residual)
        
        x = nn.relu(x + residual)
        
        return x
