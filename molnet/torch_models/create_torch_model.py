from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

import ml_collections

from molnet.torch_models import AttentionUNet

from typing import Callable, Union


def get_activation(
    activation: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Get the activation function based on the name and code."""
    if activation.lower() == "relu":
        return F.relu
    elif activation.lower() == "sigmoid":
        return F.sigmoid
    elif activation.lower() == "tanh":
        return F.tanh
    elif activation.lower() == "leaky_relu":
        return lambda x: F.leaky_relu(x, negative_slope=0.01)
    else:
        raise ValueError(f"Activation {activation} not recognized.")


def create_model(
    config: ml_collections.ConfigDict
) -> Union[nn.Module]:
    """Create a model based on the configuration."""
    if config.model_name.lower() == "torch-attention-unet":
        model = AttentionUNet(
            output_channels=config.output_channels,
            encoder_channels=config.encoder_channels,
            decoder_channels=config.decoder_channels,
            attention_channels=config.attention_channels,
            encoder_kernel_size=config.encoder_kernel_size,
            decoder_kernel_size=config.decoder_kernel_size,            
            conv_activation=get_activation(config.conv_activation),
            attention_activation=get_activation(config.attention_activation),
            return_attention_maps=config.return_attention_maps
        )

    
    return model