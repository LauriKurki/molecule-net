import jax
import jax.numpy as jnp
import flax.linen as nn

from typing import Tuple, Callable

class ResBlock(nn.Module):
    """Residual block with two convolutional layers and a skip connection."""

    channels: int
    kernel_size: Tuple[int, int, int] = (3, 3, 3)
    strides: Tuple[int, int, int] = (1, 1, 1)
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool
    ):
        residual = x
        x = nn.Conv(
            features=self.channels,
            kernel_size=self.kernel_size,
            strides=self.strides,
        )(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = self.activation(x)
        x = nn.Conv(
            features=self.channels,
            kernel_size=self.kernel_size,
            strides=self.strides,
        )(x)
        x = nn.BatchNorm(use_running_average=not training)(x)

        # Projection 
        if x.shape != residual.shape:
            residual = nn.Conv(features=self.channels, kernel_size=1)(residual)
        
        x = self.activation(x + residual)
        
        return x


class AttentionBlock3D(nn.Module):
    attention_channels: int
    kernel_size: Tuple[int, int, int] = (3, 3, 3)
    conv_activation: nn.activation = nn.relu
    attention_activation: nn.activation = nn.sigmoid

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        q: jnp.ndarray,
    ):
        # Upsample the query
        target_shape = (x.shape[0], x.shape[1], x.shape[2], x.shape[3], q.shape[-1])
        q = jax.image.resize(
            q,
            shape=target_shape,
            method='bilinear'
        )

        # Convolve the query
        q = self.conv_activation(
            nn.Conv(
                features=self.attention_channels,
                kernel_size=self.kernel_size,
            )(q)
        )

        # Convolve the input
        x = self.conv_activation(
            nn.Conv(
                features=self.attention_channels,
                kernel_size=self.kernel_size,
            )(x)
        )

        # Compute the attention
        a = self.conv_activation(x+q)
        a = self.attention_activation(
            nn.Conv(
                1,
                kernel_size=self.kernel_size
            )(a)
        )

        # Apply the attention
        y = a * x

        return y, a
