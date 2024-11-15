import tempfile
import os

import torch
torch.set_float32_matmul_precision('medium')

from configs.tests import torch_attention_test, equiv_attn_unet_test, equiv_unet_test
from molnet import train_torch as train
from configs import root_dirs

from absl.testing import parameterized, absltest
from absl import logging

ALL_CONFIGS = {
    #"unet": unet_test.get_config(),
    "attention": torch_attention_test.get_config(),
    "equiv_attention": equiv_attn_unet_test.get_config(),
    "equiv_unet": equiv_unet_test.get_config(),
}

class TrainTest(parameterized.TestCase):

    @parameterized.product(
        config_name=["equiv_unet"],
    )
    def test_train(
        self,
        config_name
    ):

        # Check if CUDA is available
        logging.info(f"Training on cuda: {torch.cuda.is_available()} -- {torch.cuda.device_count()} devices.")
        logging.info(f"Training on mps: {torch.mps.is_available()} -- {torch.mps.device_count()} devices.")

        # Get the config
        config = ALL_CONFIGS[config_name]
        config.root_dir = root_dirs.get_root_dir()

        # Create a temporary directory to store the results
        workdir = tempfile.mkdtemp()

        # Run the training
        train.train_and_evaluate(config, workdir)


if __name__ == "__main__":
    absltest.main()