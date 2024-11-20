import os
import yaml
import jax
import jax.numpy as jnp
import ml_collections
from clu import checkpoint

from molnet import train_state, utils
from molnet.models import create_model
from configs import root_dirs


def update_old_config(config):

    model_config = ml_collections.ConfigDict()

    kernel_size: int = config.model.kernel_size
    num_blocks: int = len(config.model.channels)

    model_config.encoder_kernel_size = [
        [3, 3, 3] for _ in range(num_blocks)
    ]

    model_config.decoder_kernel_size = [
        [3, 3, 3] for _ in range(num_blocks)
    ]
    
    model_config.model_name = config.model.model_name
    model_config.output_channels = config.model.output_channels
    model_config.encoder_channels = [16, 32, 64, 128]
    model_config.decoder_channels = [128, 64, 32, 16]
    model_config.attention_channels = [16, 16, 16, 16]
    model_config.conv_activation = "relu"
    model_config.attention_activation = "sigmoid"
    model_config.return_attention_maps = True

    return model_config


def load_from_workdir(
    workdir: str,
    return_attention: bool,
    old: bool = False
):
    # Load the model config
    with open(os.path.join(workdir, "config.yaml"), "rt") as f:
        config = yaml.unsafe_load(f)
    config = ml_collections.ConfigDict(config)
    config.root_dir = root_dirs.get_root_dir()
    config.model.return_attention_maps = return_attention

    if old:
        model_config = update_old_config(config)
    else:
        model_config = config.model

    # Create the model
    model = create_model(model_config)

    # Load from the checkpoint
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    ckpt = checkpoint.Checkpoint(checkpoint_dir)
    restored_state = ckpt.restore(state=None)['state']

    # Create the optimizer and the evaluation state
    apply_fn = model.apply
    tx = utils.create_optimizer(config)

    # Create the evaluation state for predictions
    state = train_state.EvaluationState.create(
        apply_fn=apply_fn,
        params=restored_state['params'],
        batch_stats=restored_state['batch_stats'],
        tx=tx,
    )
    state = jax.tree_util.tree_map(jnp.asarray, state)

    return state, config
