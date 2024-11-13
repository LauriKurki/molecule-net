import tempfile
import os

import torch
torch.set_float32_matmul_precision('medium')
import tensorflow as tf

from configs.tests import torch_attention_test
from molnet import train_torch as train
from configs import root_dirs

from absl.testing import parameterized, absltest
from absl import logging

ALL_CONFIGS = {
    #"unet": unet_test.get_config(),
    "attention": torch_attention_test.get_config(),
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
        tf.config.experimental.set_visible_devices([], 'GPU')
        
        # Check if CUDA is available
        logging.info(f"Training on cuda: {torch.cuda.is_available()} -- {torch.cuda.device_count()} devices.")

        # Get the config
        config = ALL_CONFIGS[config_name]
        config.root_dir = root_dirs.get_root_dir()

        # Create a temporary directory to store the results
        workdir = tempfile.mkdtemp()

        # Run the training
        train.train_and_evaluate(config, workdir)


if __name__ == "__main__":
    absltest.main()