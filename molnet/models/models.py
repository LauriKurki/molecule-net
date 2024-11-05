import jax
import jax.numpy as jnp
import flax.linen as nn

from molnet.models.layers import ResBlock, AttentionBlock3D

from typing import Sequence

class UNet(nn.Module):

    output_channels: int
    filters: Sequence[int]
    kernel_size: Sequence[int]

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool,
    ) -> jnp.ndarray:
        # Downward path
        skips = []
        for i, (f, k) in enumerate(zip(self.filters, self.kernel_size)):
            x = ResBlock(
                f, (k, k, k), name=f"encoder_{i}"
            )(x, training)
            skips.append(x)

            x = nn.max_pool(x, window_shape=(2, 2, 1), strides=(2, 2, 1))

        # Bottom path
        x = ResBlock(
            self.filters[-1],
            (self.kernel_size[-1],)*3,
            name="bottom"
        )(x, training)

        # Upward path
        for i, (f, k) in enumerate(zip(reversed(self.filters[:-1]), reversed(self.kernel_size[:-1]))):
            target_shape = (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3], x.shape[4])
            x = jax.image.resize(
                x,
                shape=target_shape,
                method='bilinear'
            )
            skip_x = skips.pop()
            x = jnp.concatenate([x, skip_x], axis=-1)
            
            x = ResBlock(
                f, (k, k, k), name=f"decoder_{i}"
            )(x, training)


        target_shape = (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3], x.shape[4])
        x = jax.image.resize(
            x,
            shape=target_shape,
            method='bilinear'
        )

        # Resblock before output
        x = ResBlock(
            self.filters[0],
            (self.kernel_size[0],)*3,
            name="final"
        )(x, training)

        # Output
        x = nn.Conv(
            self.output_channels,
            kernel_size=(1, 1, 1),
            padding='SAME',
            name="output"
        )(x)

        return x


class AttentionUNet(nn.Module):
    output_channels: int
    channels: Sequence[int]
    attention_channels: Sequence[int]
    kernel_size: Sequence[int]
    return_attention_maps: bool = False

    @nn.compact
    def __call__(self, x, training: bool):
        attention_maps = []
        skips = []
        for i, (f, k) in enumerate(zip(self.channels, self.kernel_size)):
            x = ResBlock(
                f, (k, k, k), name=f"encoder_{i}"
            )(x, training)
            skips.append(x)

            x = nn.max_pool(x, window_shape=(2, 2, 1), strides=(2, 2, 1))

        # Bottom path
        x = ResBlock(
            self.channels[-1],
            (self.kernel_size[-1],)*3,
            name="bottom"
        )(x, training)

        # Upward path
        for i, (f, k) in enumerate(zip(reversed(self.channels), reversed(self.kernel_size))):
            # Attention
            a, map_i = AttentionBlock3D(
                attention_channels=self.attention_channels[i],
                kernel_size=(3, 3, 3),
                conv_activation=nn.relu,
                attention_activation=nn.sigmoid,
                name=f"attention_{i}"
            )(skips.pop(), x)
            attention_maps.append(map_i)

            target_shape = (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3], x.shape[4])
            x = jax.image.resize(
                x,
                shape=target_shape,
                method='bilinear'
            )

            x = jnp.concatenate([x, a], axis=-1)
            x = ResBlock(
                f, (k, k, k), name=f"decoder_{i}"
            )(x, training)

        # Output
        x = nn.Conv(
            self.output_channels,
            kernel_size=(1, 1, 1),
            padding='SAME',
            name="output"
        )(x)


        if self.return_attention_maps:
            return x, attention_maps
        else:
            return x
