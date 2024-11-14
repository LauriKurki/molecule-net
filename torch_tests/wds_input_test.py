import time

from molnet.data import input_pipeline_wds

from absl.testing import absltest

from configs import root_dirs
from configs.tests import torch_attention_test


class TestInputPipeline(absltest.TestCase):
    def test_input(self):
        config = torch_attention_test.get_config()
        config.root_dir = root_dirs.get_root_dir()
        config.num_workers = 8

        print(config)

        datasets = input_pipeline_wds.get_datasets(config)

        trainset = datasets["train"]

        times = []
        t0 = time.perf_counter()
        for i, batch in enumerate(trainset):

            x, y, xyz = batch

            print(f"Shapes: x: {x.shape}, y: {y.shape}, xyz: {xyz.shape}")

            t1 = time.perf_counter()
            times.append(t1 - t0)

            print(f"\n Time to get batch: {(t1 - t0)*1e3:.2f} ms")
            t0 = time.perf_counter()

        print(f"Average time to get batch: {sum(times)/len(times)*1e3:.2f} ms")

if __name__ == '__main__':
    absltest.main()