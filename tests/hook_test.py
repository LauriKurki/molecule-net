from absl.testing import absltest
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from clu import metric_writers

import optax

from molnet.models import create_model
from molnet.data import input_pipeline
from molnet import train_state, loss, hooks

from configs.tests import unet_test


@jax.jit
def predict_step(state, batch):
    inputs, targets = batch['images'], batch['atom_map']
    preds = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        inputs,
        training=False,
    )
    preds_z = preds.shape[-2]
    target = targets[..., -preds_z:, :]
    loss_by_image = jnp.mean(
        (preds - target) ** 2,
        axis=(1, 2, 3, 4),
    )
    return inputs, target, preds, loss_by_image

def predict_with_state(state, dataset, num_batches=1):
    losses = []
    preds = []
    inputs = []
    targets = []
    
    for i in range(num_batches):
        batch = next(dataset)
        (
            batch_inputs, batch_targets, batch_preds, batch_loss
        ) = predict_step(state, batch)
        inputs.append(batch_inputs)
        targets.append(batch_targets)
        preds.append(batch_preds)
        losses.append(batch_loss)

    inputs = jnp.concatenate(inputs)
    targets = jnp.concatenate(targets)
    preds = jnp.concatenate(preds)
    losses = jnp.concatenate(losses)

    return inputs, targets, preds, losses


def get_config():
    config = unet_test.get_config()
    config.root_dir = '/l/data/molnet/atom_maps/'
    return config

class HookTest(absltest.TestCase):
    def setUp(self):
        self.config = get_config()
        rng = jax.random.PRNGKey(0)
        datarng, rng = jax.random.split(rng)
        datasets = input_pipeline.get_datasets(datarng, self.config)

        model = create_model(self.config)
        x_init = next(datasets['train'])['images']
        variables = model.init(rng, x_init, training=True)
        params = variables['params']
        batch_stats = variables['batch_stats']

        self.state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            batch_stats=batch_stats,
            tx=optax.adamw(1e-3),
            best_params=params,
            metrics_for_best_params={},
            step_for_best_params=0,
        )

        workdir = './test_workdir'
        self.workdir = workdir

        writer = metric_writers.SummaryWriter(self.workdir)
        self.hook = hooks.PredictionHook(
            workdir=workdir,
            predict_fn=lambda state, num_batches: predict_with_state(
                state,
                datasets['val'],
                num_batches,
            ),
            writer=writer,
        )

    def test_prediction_hook(self):
        self.hook(self.state, 1, False)
        self.assertTrue(os.path.exists(self.workdir))


if __name__ == '__main__':
    absltest.main()
