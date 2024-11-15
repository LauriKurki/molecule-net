from torch import nn
from torch.nn import functional as F

import escnn
from escnn import nn as enn

from typing import Union


def _get_padding(kernel_size):
    padding = [(kernel_size-1) // 2]*2
    return tuple(padding)


class EqConv2d(enn.EquivariantModule):
    """
    Equivariant Pytorch 2D convolutional block.

    Args:
        gspace (`escnn.gspaces.GSpace`): The space that describes
            the symmetries the features should have.
        ch_in (int): number of independent feature fields in the input.
        ch_out (int): number of independent feature fields in the output.
        n_layers (int): number of convolutional layers in the block.
        kernel_size (int): Size of convolution kernel.
        padding (int): Amount of padding applied to the input.
        scalar_input (boolean): Input transforms as a trivial representation.
            True for first layer of model assuming image input.
        scalar_output (boolean): Output transforms as a trivial representation.
            True for last layer of model assuming image output.
        res_connection (boolean): Applies residual connection over the block.
    """
    def __init__(self,
                 gspace: escnn.gspaces.GSpace,
                 ch_in: int,
                 ch_out: int,
                 n_layers: int = 2,
                 kernel_size: int = 3,
                 padding: Union[str, int] = 'same',
                 scalar_input: bool = False,
                 scalar_output: bool = False,
                 res_connection: bool = True):

        assert n_layers > 0
        assert isinstance(scalar_input, bool)
        assert isinstance(scalar_output, bool)

        super(EqConv2d, self).__init__()

        self.res_connection = res_connection

        if isinstance(padding, str):
            if padding == 'same':
                padding = _get_padding(kernel_size)
            else:
                raise ValueError('invalid padding')
        elif isinstance(padding, int):
            padding = padding

        if scalar_input:
            c_in = enn.FieldType(gspace, ch_in*[gspace.trivial_repr])
        else:
            c_in = enn.FieldType(gspace, ch_in*[gspace.regular_repr])

        if scalar_output:
            c_out = enn.FieldType(gspace, ch_out*[gspace.trivial_repr])
        else:
            c_out = enn.FieldType(gspace, ch_out*[gspace.regular_repr])

        layer_1 = enn.SequentialModule(
            enn.R2Conv(c_in, c_out, kernel_size=kernel_size, padding=padding),
            enn.InnerBatchNorm(c_out),
            enn.ReLU(c_out))

        layer_n = enn.SequentialModule(
            enn.R2Conv(c_out, c_out, kernel_size=kernel_size, padding=padding),
            enn.InnerBatchNorm(c_out),
            enn.ReLU(c_out))

        self.eqconvs = nn.ModuleList([layer_1] + (n_layers-1)*[layer_n])

        # if input and output have different channels, do 1x1 convolution
        if res_connection and ch_in != ch_out:
            self.res_conv = enn.R2Conv(c_in, c_out, kernel_size=1, padding=0)
        else:
            self.res_conv = None

    def forward(self, x_in: enn.GeometricTensor):
        x = x_in

        for conv in self.eqconvs:
            x = conv(x)

        if self.res_connection:
            if self.res_conv:
                x = x + self.res_conv(x_in)
            else:
                x = x + x_in

        return x

