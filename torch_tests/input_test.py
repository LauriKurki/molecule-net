import jax
import torch
import ml_collections
import time

from absl.testing import absltest

from configs.tests import unet_test, attention_test
from molnet.data import input_pipeline
from configs import root_dirs


class TestInputPipeline(absltest.TestCase):
    def test_input_pipeline(self):
        config = attention_test.get_config()
        config.root_dir = root_dirs.get_root_dir()

        print(config)

        rng = jax.random.PRNGKey(0)
        datarng, rng = jax.random.split(rng)
        datasets = input_pipeline.get_pseudodatasets(datarng, config)

        trainset = datasets["train"]

        times = []
        for i in range(10):

            t0 = time.perf_counter()
            batch = next(trainset)
            batch = jax.tree_util.tree_map(torch.from_numpy, batch)
            
            # Print the shapes of each item in the batch.
            for key, value in batch.items():
                print(key, value.shape, end=", ")

            t1 = time.perf_counter()
            times.append(t1 - t0)
            print(f"\n Time to get batch: {(t1 - t0)*1e3:.2f} ms")

        print(f"Average time to get batch: {sum(times)/len(times)*1e3:.2f} ms")

if __name__ == '__main__':
    absltest.main()