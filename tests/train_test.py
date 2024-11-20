import tempfile
import os
#os.environ['XLA_FLAGS'] = "--xla_force_host_platform_device_count=4"
#os.environ['JAX_PLATFORMS']='cpu'

import jax
#import tensorflow as tf

from configs.tests import unet_test, attention_test
from molnet import train
from configs import root_dirs

from absl.testing import parameterized, absltest
from absl import logging

ALL_CONFIGS = {
    #"unet": unet_test.get_config(),
    "attention": attention_test.get_config(),
}

class TrainTest(parameterized.TestCase):

    @parameterized.product(
        config_name=list(ALL_CONFIGS.keys())
    )
    def test_train(
        self,
        config_name
    ):

        #Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
        # it unavailable to JAX.
        #tf.config.experimental.set_visible_devices([], 'GPU')

        logging.info(f"Running test on devices: {jax.devices()}")

        # Enable NaN and Inf checking
        jax.config.update("jax_debug_nans", True)
        jax.config.update("jax_debug_infs", True)

        # Get the config
        config = ALL_CONFIGS[config_name]
        config.root_dir = root_dirs.get_root_dir()

        # Create a temporary directory to store the results
        workdir = tempfile.mkdtemp()

        # Run the training
        train.train_and_evaluate(config, workdir)


if __name__ == "__main__":
    absltest.main()