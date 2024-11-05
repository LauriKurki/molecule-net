import tempfile
import os
#os.environ['XLA_FLAGS'] = "--xla_force_host_platform_device_count=4"
#os.environ['JAX_PLATFORMS']='cpu'

import jax
import tensorflow as tf

from molnet import train
from configs import root_dirs, test

from absl.testing import absltest

class TrainTest(absltest.TestCase):

    def test_train(self):

      #Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
      # it unavailable to JAX.
      tf.config.experimental.set_visible_devices([], 'GPU')

      print("Running test on devices: ", jax.devices())

      # Enable NaN and Inf checking
      jax.config.update("jax_debug_nans", True)
      jax.config.update("jax_debug_infs", True)

      # Get the config
      config = test.get_config()
      config.root_dir = root_dirs.get_root_dir()

      # Create a temporary directory to store the results
      workdir = tempfile.mkdtemp()

      # Run the training
      train.train(config, workdir)


if __name__ == "__main__":
    absltest.main()