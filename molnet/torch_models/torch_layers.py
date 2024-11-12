import torch
from torch import nn
from torch.nn import functional as F

from typing import Union, Tuple, Callable, Sequence

class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]],
        activation: Callable[[torch.Tensor], torch.Tensor] = F.relu
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding=1)
        self.res_conv = nn.Conv3d(in_channels, out_channels, 1, stride)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.act = activation

    def forward(self, x):
        res = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if x.shape != res.shape:
            res = self.res_conv(res)

        x = self.act(x + res)
        return x
    
class AttentionBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        query_channels: int,
        attention_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]],
        conv_activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        attention_activation: Callable[[torch.Tensor], torch.Tensor] = F.sigmoid
    ):
        super().__init__()
        self.x_conv = nn.Conv3d(in_channels, attention_channels, kernel_size, stride, padding=1)
        self.q_conv = nn.Conv3d(query_channels, attention_channels, kernel_size, stride, padding=1)
        self.a_conv = nn.Conv3d(attention_channels, 1, kernel_size, stride, padding=1)

        self.conv_activation = conv_activation
        self.attention_activation = attention_activation

    def forward(self, x, q):
        q = F.interpolate(q, size=x.size()[2:], mode="trilinear", align_corners=False)

        # convolve query
        q = self.conv_activation(self.q_conv(q))

        # convolve x and sum
        x = self.conv_activation(self.x_conv(x))
        a = self.conv_activation(x+q)

        a = self.attention_activation(self.a_conv(a))
        x = x * a

        return x, a
