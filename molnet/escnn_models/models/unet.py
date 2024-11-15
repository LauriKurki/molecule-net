import torch
import torch.nn as nn

from escnn.gspaces import GSpace
import escnn.nn as enn

from molnet import escnn_models

from typing import List, Union, Sequence, Callable, Optional


class EquivariantUNet(nn.Module):
    r"""
    Equivariant 3D UNet implementation.

    Args:
        gspace (`escnn.gspaces.GSpace`): The space that describes
            the symmetries the features should have.
        encoder_channels (list of int): Number of channels
            in each encoder block output.
        decoder_channels (list of int): Number of channels
            in each decoder block input.
        output_channels (int): Number of channels in the output.
        kernel_size (int): Size of the kernel used in convolutions.
        padding (str or int): 'same' for preserving spatial size or int for
            amount of padding used in convolutions.
    """
    def __init__(
        self,
        gspace: GSpace,
        output_channels: int,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
        encoder_kernel_size: Sequence[List[int]],
        decoder_kernel_size: Sequence[List[int]],
        conv_activation: Callable[[torch.Tensor], torch.Tensor],
        input_size: Optional[int] = (128, 128, 10),
    ):
        super(EquivariantUNet, self).__init__()


        self.gspace = gspace
        num_blocks = len(encoder_channels)

        # Input type
        self.in_type = enn.FieldType(
            self.gspace,
            [self.gspace.trivial_repr]
        )

        # Mask input since corners are moved outside of the grid during rotations
        self.mask = escnn_models.MaskModule3D(
            self.in_type,
            S=input_size,
            margin=5,
            sigma=20
        )

        # Conv for making input non-scalar
        self.in_conv = enn.R3Conv(
            self.in_type,
            enn.FieldType(
                self.gspace,
                [self.gspace.regular_repr]*encoder_channels[0]
            ),
            kernel_size=1
        )

        # Encoder blocks
        encoder_in_channels = [encoder_channels[0]] + encoder_channels[:-1]
        encoder_out_channels = encoder_channels
        self.encoder = nn.ModuleList()
        skip_channels = []
        self.pools = nn.ModuleList()
        for i in range(num_blocks):
            encoder_i = escnn_models.EqConv3d(
                gspace=self.gspace,
                ch_in=encoder_in_channels[i],
                ch_out=encoder_out_channels[i],
                kernel_size=encoder_kernel_size[i],
                scalar_input=False,
                scalar_output=False,
                activation=conv_activation
            )
            pool_i = escnn_models.NormMaxPool3D(
                in_type=encoder_i.out_type,
                kernel_size=(2, 2, 1),
            )
            self.encoder.append(encoder_i)
            skip_channels.append(encoder_out_channels[i])
            self.pools.append(pool_i)

        # Bottom block
        self.bottom_block = escnn_models.EqConv3d(
            gspace=self.gspace,
            ch_in=encoder_channels[-1],
            ch_out=encoder_channels[-1],
            kernel_size=encoder_kernel_size[-1],
            scalar_input=False,
            scalar_output=False,
            activation=conv_activation
        )

        upsamling_in_type = self.bottom_block.out_type
        self.upsamplings = nn.ModuleList()

        # Decoder blocks
        decoder_in_channels = [encoder_channels[-1]] + decoder_channels[:-1]
        decoder_out_channels = decoder_channels
        self.decoder = nn.ModuleList()
        for i in range(num_blocks):
            up_i = enn.R3Upsampling(
                in_type=upsamling_in_type,
                scale_factor=(2, 2, 1)
            )
            decoder_i = escnn_models.EqConv3d(
                gspace=self.gspace,
                ch_in=decoder_in_channels[i]+skip_channels[-i-1],
                ch_out=decoder_out_channels[i],
                kernel_size=decoder_kernel_size[i],
                scalar_input=False,
                scalar_output=False,
                activation=conv_activation
            )
            self.upsamplings.append(up_i)
            self.decoder.append(decoder_i)

            upsamling_in_type = decoder_i.out_type

        # Output type
        self.out_type = enn.FieldType(
            self.gspace,
            [self.gspace.trivial_repr]*output_channels
        )
        self.out_conv = enn.R3Conv(
            self.decoder[-1].out_type,
            self.out_type,
            kernel_size=1,
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # wrap input into geometric tensor
        x = self.in_type(x)
        
        # mask input
        x = self.mask(x)

        # convolve input
        x = self.in_conv(x)

        # Encoder
        skips = []
        for block, pooling in zip(self.encoder, self.pools):
            x = block(x)
            skips.append(x)

            x = pooling(x)

        # Bottom
        x = self.bottom_block(x)

        # Decoder
        for block, up in zip(self.decoder, self.upsamplings):
            x = up(x)
            skip = skips.pop()
            x = enn.tensor_directsum([x, skip])
            x = block(x)

        # Output
        x = self.out_conv(x)

        # unwrap output from geometric tensor
        x = x.tensor
        return x
