import time
import jax
import jax.numpy as jnp

from absl.testing import absltest

from configs.tests import attention_test
from molnet.data import input_pipeline_seg
from configs import root_dirs


class TestInputPipeline(absltest.TestCase):
    def test_input_pipeline(self):
        config = attention_test.get_config()
        config.root_dir = root_dirs.get_root_dir(config.dataset)

        print(config)

        datasets = input_pipeline_seg.get_datasets(config)
        trainloader = datasets["train"]

        times = []
        for i in range(100):

            t0 = time.perf_counter()
            batch = next(trainloader)
            batch = jax.tree_util.tree_map(lambda x: jnp.array(x, dtype=jnp.bfloat16), batch)
            t1 = time.perf_counter()
            
            # Print the shapes of each item in the batch.
            for key, value in batch.items():
                print(key, value.dtype, value.shape, end=" -- ")

            print(f"\n Time to get batch: {(t1 - t0)*1e3:.2f} ms")
            time.sleep(0.2)
            if i < 5:
                continue
            times.append(t1 - t0)

        print(f"Average time to get batch: {sum(times)/len(times)*1e3:.2f} ms")

if __name__ == '__main__':
    absltest.main()
