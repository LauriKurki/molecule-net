import time
import tensorflow as tf

from molnet.data import utils
from molnet.data import input_pipeline_fdbm

from absl.testing import absltest

from configs.tests import fdbm_test


class TestInputPipeline(absltest.TestCase):
    def test_input(self):
        config = fdbm_test.get_config()
        config.dataset = "SIN-AFM-FDBM-tf"
        config.root_dir = "/u/79/kurkil1/unix/work/molnet/data/SIN-AFM-FDBM-tf"
        #config.root_dir = root_dirs.get_root_dir(config.dataset)
        config.num_workers = 8

        config.batch_size = 12

        # Set the random number generator seeds
        tf.random.set_seed(config.rng_seed)
        tf.random.set_global_generator(tf.random.Generator.from_seed(config.rng_seed))

        datasets = input_pipeline_fdbm.get_datasets(config)

        trainloader = datasets["train"]

        times = []
        for i in range(50):
            t0 = time.perf_counter()
            batch = next(trainloader)
            t1 = time.perf_counter()

            #x, sw, xyz = batch["images"], batch["sw"], batch["xyz"]
            x, sw, xyz, target = batch["images"], batch["sw"], batch["xyz"], batch["atom_map"]

            print(f"shapes: {x.shape}, {sw.shape}, {xyz.shape} {target.shape}")
            print(f"dtypes: {x.dtype}, {sw.dtype}, {xyz.dtype} {target.dtype}")
            print(f"means: {x.mean()}, {sw.mean()}, {xyz.mean()} {target.mean()}")

            if i < 5: continue
            times.append(t1 - t0)

            print(f"\n Time to get batch: {(t1 - t0)*1e3:.2f} ms")
            time.sleep(0.5)


        print(f"Average time to get batch: {sum(times)/len(times)*1e3:.2f} ms")


if __name__ == '__main__':
    absltest.main()
