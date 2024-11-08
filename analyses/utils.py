import os
import yaml
import jax
import jax.numpy as jnp
import ml_collections
from clu import checkpoint

from molnet import train_state, utils
from molnet.models import create_model
from configs import root_dirs

def load_from_workdir(
    workdir: str,
    return_attention: bool
):
    # Load the model config
    with open(os.path.join(workdir, "config.yaml"), "rt") as f:
        config = yaml.unsafe_load(f)
    config = ml_collections.ConfigDict(config)
    config.root_dir = root_dirs.get_root_dir()
    config.model.return_attention_maps = return_attention

    # Create the model
    model = create_model(config.model)

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
