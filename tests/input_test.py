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

        datasets = input_pipeline.get_pseudodatasets(config)
        trainloader = datasets["train"]

        times = []
        for i in range(100):

            t0 = time.perf_counter()
            batch = next(trainloader)
            t1 = time.perf_counter()
            
            # Print the shapes of each item in the batch.
            for key, value in batch.items():
                print(key, type(value), value.shape, end=" -- ")

            print(f"\n Time to get batch: {(t1 - t0)*1e3:.2f} ms")
            time.sleep(0.2)
            if i < 5:
                continue
            times.append(t1 - t0)

        print(f"Average time to get batch: {sum(times)/len(times)*1e3:.2f} ms")

if __name__ == '__main__':
    absltest.main()
