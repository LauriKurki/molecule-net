import jax
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
        datasets = input_pipeline.get_datasets(datarng, config)

        trainset = datasets["train"]

        for i in range(10):

            t0 = time.perf_counter()
            batch = next(trainset)
            
            # Print the shapes of each item in the batch.
            for key, value in batch.items():
                print(key, value.shape, end=", ")

            t1 = time.perf_counter()
            print(f"\n Time to get batch: {(t1 - t0)*1e3:.2f} ms")

if __name__ == '__main__':
    absltest.main()