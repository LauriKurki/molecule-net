import jax
import ml_collections
import time

from absl.testing import absltest

from configs.tests import unet_test
from molnet.data import input_pipeline
from configs import root_dirs


class TestInputPipeline(absltest.TestCase):
    def test_input_pipeline(self):
        config = unet_test.get_config()
        config.root_dir = root_dirs.get_root_dir()

        print(config)

        rng = jax.random.PRNGKey(0)
        datarng, rng = jax.random.split(rng)
        datasets = input_pipeline.get_datasets(datarng, config)

        trainset = datasets["train"]

        for i in range(10):

            t0 = time.perf_counter()
            batch = next(trainset)
            image = batch["images"]
            atom_map = batch["atom_map"]
            #xyz = batch["xyz"]

            print(f"Batch {i}: {image.shape}, {atom_map.shape}")
            print(f"Time taken: {(time.perf_counter() - t0)*1000:.2f} ms")


if __name__ == '__main__':
    absltest.main()