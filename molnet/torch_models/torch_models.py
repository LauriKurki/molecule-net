import torch
from torch import nn

from typing import Sequence, Callable, List

from molnet.torch_models.torch_layers import ResBlock, AttentionBlock3D

class AttentionUNet(nn.Module):
    def __init__(
        self,
        output_channels: int,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
        attention_channels: Sequence[int],
        encoder_kernel_size: Sequence[List[int]],
        decoder_kernel_size: Sequence[List[int]],
        conv_activation: Callable[[torch.Tensor], torch.Tensor],
        attention_activation: Callable[[torch.Tensor], torch.Tensor],
        return_attention_maps: bool = False
    ):
        super().__init__()
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.attention_channels = attention_channels
        self.encoder_kernel_size = encoder_kernel_size
        self.decoder_kernel_size = decoder_kernel_size
        self.conv_activation = conv_activation
        self.attention_activation = attention_activation
        self.return_attention_maps = return_attention_maps

        # Define encoder blocks
        self.encoder_blocks = nn.ModuleList(
            [
                ResBlock(
                    encoder_channels[i],
                    (encoder_kernel_size[i][0], encoder_kernel_size[i][1], encoder_kernel_size[i][2]),
                    activation=conv_activation,
                    stride=1
                )
                for i in range(len(encoder_channels))
            ]
        )

        # Define bottom block
        self.bottom_block = ResBlock(
            encoder_channels[-1],
            (encoder_kernel_size[-1][0], encoder_kernel_size[-1][1], encoder_kernel_size[-1][2]),
            activation=conv_activation,
            stride=1
        )

        # Define attention blocks
        self.attention_blocks = nn.ModuleList(
            [
                AttentionBlock3D(
                    list(reversed(encoder_channels))[i],
                    attention_channels[i],
                    (3, 3, 3),
                    conv_activation,
                    attention_activation
                )
                for i in range(len(attention_channels))
            ]
        )

        # Define decoder blocks
        self.decoder_blocks = nn.ModuleList(
            [
                ResBlock(
                    decoder_channels[i],
                    (decoder_kernel_size[i][0], decoder_kernel_size[i][1], decoder_kernel_size[i][2]),
                    activation=conv_activation,
                    stride=1
                )
                for i in range(len(decoder_channels))
            ]
        )

        # Define output
        self.output = nn.Conv3d(
            decoder_channels[-1],
            output_channels,
            kernel_size=(1, 1, 1),
            padding=(0, 0, 0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []

        # Encoder path
        for block in self.encoder_blocks:
            # Apply conv block and save the skip connection + downsample
            x = block(x)
            skips.append(x)
        
            x = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))(x)

        # Bottom path
        x = self.bottom_block(x)

        # Upward path
        for i, (attn, decoder_block) in enumerate(zip(self.attention_blocks, self.decoder_blocks)):
            # Attention
            a, _ = attn(skips.pop(), x)

            # Upsample
            target_shape = (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3], x.shape[4])
            x = nn.functional.interpolate(x, size=target_shape[1:], mode='trilinear', align_corners=True)
            
            # Concatenate attention and x + apply conv block
            x = torch.cat([x, a], dim=1)
            x = decoder_block(x)

        # Output
        x = self.output(x)

        return x
