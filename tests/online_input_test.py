import time
import numpy as np
import matplotlib.pyplot as plt

from molnet.data import utils
from molnet.data import input_pipeline_online

from absl.testing import absltest

from configs import root_dirs
from configs.tests import attention_test


class TestInputPipeline(absltest.TestCase):
    def test_input(self):
        config = attention_test.get_config()
        config.root_dir = root_dirs.get_root_dir()
        config.num_workers = 8

        config.batch_size = 12
        config.train_molecules = (0, 50000)
        config.val_molecules = (50000, 60000)
        config.interpolate_input_z = 20

        print(config)

        datasets = input_pipeline_online.get_datasets(config)

        trainloader = datasets["train"]

        times = []
        for i in range(100):
            t0 = time.perf_counter()
            batch = next(trainloader)
            t1 = time.perf_counter()

            #x, sw, xyz = batch["images"], batch["sw"], batch["xyz"]
            x, sw, xyz, target = batch["images"], batch["sw"], batch["xyz"], batch["atom_map"]

            print(f"shapes: {x.shape}, {sw.shape}, {xyz.shape} {target.shape}")
            print(f"dtypes: {x.dtype}, {sw.dtype}, {xyz.dtype} {target.dtype}")     

            if i < 5: continue
            times.append(t1 - t0)

            print(f"\n Time to get batch: {(t1 - t0)*1e3:.2f} ms")
            time.sleep(0.1)


        print(f"Average time to get batch: {sum(times)/len(times)*1e3:.2f} ms")


if __name__ == '__main__':
    absltest.main()