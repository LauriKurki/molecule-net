from absl.testing import absltest
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from clu import metric_writers

import optax

from molnet.models import create_model
from molnet.data import input_pipeline
from molnet import train_state, loss, hooks, train

from configs import root_dirs
from configs.tests import attention_test


def get_config():
    config = attention_test.get_config()
    config.root_dir = root_dirs.get_root_dir()
    return config

class HookTest(absltest.TestCase):
    def setUp(self):
        self.config = get_config()
        rng = jax.random.PRNGKey(0)
        datarng, rng = jax.random.split(rng)
        datasets = input_pipeline.get_datasets(datarng, self.config)

        model = create_model(self.config.model)
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
            predict_fn=lambda state, num_batches: train.predict_with_state(
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
