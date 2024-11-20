import time

from molnet.data import input_pipeline_wds

from absl.testing import absltest

from configs import root_dirs
from configs.tests import attention_test


class TestInputPipeline(absltest.TestCase):
    def test_input(self):
        config = attention_test.get_config()
        config.root_dir = root_dirs.get_root_dir()
        config.num_workers = 8

        config.batch_size = 8
        config.train_molecules = (0, 50000)
        config.val_molecules = (50000, 60000)

        print(config)

        datasets = input_pipeline_wds.get_datasets(config)

        trainloader = iter(datasets["train"])

        times = []
        for i in range(100):
            t0 = time.perf_counter()
            try:
                batch = next(trainloader)
            except StopIteration:
                t1 = time.perf_counter()
                print(f"Time to exit / restart: {(t1 - t0)*1e3:.2f} ms")
                trainloader = iter(datasets["train"])
                continue
                #break
            t1 = time.perf_counter()

            if isinstance(batch, dict):
                print(batch.keys())
                x, y, xyz = batch["images"], batch["atom_map"], batch["xyz"]
            else:
                x, y, xyz = batch

            print(f"Shapes: x: {x.shape}, y: {y.shape}, xyz: {xyz.shape}")
            print(f"dtypes: x: {x.dtype}, y: {y.dtype}, xyz: {xyz.dtype}")
            print(f"x min: {x.min()}, x max: {x.max()}, x mean: {x.mean()}")

            if i < 5: continue
            times.append(t1 - t0)

            print(f"\n Time to get batch: {(t1 - t0)*1e3:.2f} ms")
            time.sleep(0.2)


        print(f"Average time to get batch: {sum(times)/len(times)*1e3:.2f} ms")

if __name__ == '__main__':
    absltest.main()
