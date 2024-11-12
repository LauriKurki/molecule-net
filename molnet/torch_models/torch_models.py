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

        encoder_in_channels = [1] + encoder_channels[:-1] # [1, F1, F2]
        encoder_out_channels = encoder_channels           # [F1, F2, F3]

        # Define encoder blocks
        self.encoder_blocks = nn.ModuleList(
            [
                ResBlock(
                    in_channels=encoder_in_channels[i],
                    out_channels=encoder_out_channels[i],
                    kernel_size=(
                        encoder_kernel_size[i][0],
                        encoder_kernel_size[i][1],
                        encoder_kernel_size[i][2]
                    ),
                    stride=(1, 1, 1),
                    activation=conv_activation,
                )
                for i in range(len(encoder_channels))
            ]
        )

        # Define bottom block
        self.bottom_block = ResBlock(
            in_channels=encoder_channels[-1],  # [F3]
            out_channels=encoder_channels[-1], # [F3]
            kernel_size=(
                encoder_kernel_size[-1][0],
                encoder_kernel_size[-1][1],
                encoder_kernel_size[-1][2]
            ),
            stride=(1, 1, 1),
            activation=conv_activation,
        )

        attention_x_channels = decoder_channels # [F3, F2, F1]
        attention_q_channels = decoder_channels # [F3, F2, F1]

        # Define attention blocks
        self.attention_blocks = nn.ModuleList(
            [
                AttentionBlock3D(
                    in_channels=attention_x_channels[i],
                    query_channels=attention_q_channels[i],
                    attention_channels=attention_channels[i],
                    kernel_size=(3, 3, 3),
                    stride=(1, 1, 1),
                    conv_activation=conv_activation,
                    attention_activation=attention_activation
                )
                for i in range(len(attention_channels))
            ]
        )

        # Define decoder input channels [F3+A1, F2+A2, F1+A3]
        decoder_in_channels = [
            decoder_channels[i] + attention_channels[i]
            for i in range(len(decoder_channels))
        ]
        # Decoder out channels[F2, F1, F1]
        decoder_out_channels = decoder_channels[1:] + [decoder_channels[-1]]

        # Define decoder blocks
        self.decoder_blocks = nn.ModuleList(
            [
                ResBlock(
                    in_channels=decoder_in_channels[i],
                    out_channels=decoder_out_channels[i],
                    kernel_size=(
                        decoder_kernel_size[i][0],
                        decoder_kernel_size[i][1],
                        decoder_kernel_size[i][2]
                    ),
                    stride=(1, 1, 1),
                    activation=conv_activation,
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

    def forward(
        self,
        x: torch.Tensor,
        return_attention_maps: bool = False
    ) -> torch.Tensor:
        skips = []

        # Encoder path
        for i, block in enumerate(self.encoder_blocks):
            # Apply conv block and save the skip connection + downsample
            x = block(x)
            skips.append(x)

            x = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))(x)

        # Bottom path
        x = self.bottom_block(x)

        # Upward path
        maps = []
        for i, (attn, decoder_block) in enumerate(zip(self.attention_blocks, self.decoder_blocks)):
            # Attention
            attn_x = skips.pop()
            a, attention_map = attn(attn_x, x)
            maps.append(attention_map)

            # Upsample
            target_shape = (x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3] * 2, x.shape[4])
            x = nn.functional.interpolate(x, size=target_shape[2:], mode='trilinear', align_corners=True)
            
            # Concatenate attention and x + apply conv block
            x = torch.cat([x, a], dim=1)
            x = decoder_block(x)

        # Output
        x = self.output(x)

        if return_attention_maps:
            return x, maps
        else:
            return x
