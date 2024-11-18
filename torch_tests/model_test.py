import os
import time

from absl.testing import parameterized, absltest
from absl import logging

import torch

from configs.tests import torch_attention_test, equiv_attn_unet_test, equiv_unet_test

from molnet.torch_models import create_model

ALL_CONFIGS = {
    #"unet": unet_test.get_config(),
    "attention": torch_attention_test.get_config(),
    "equiv_unet": equiv_unet_test.get_config(),
    "equiv_attention": equiv_attn_unet_test.get_config(),
}

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
else:
    device = torch.device("cpu")


class ModelTest(parameterized.TestCase):
        
        def setUp(self):

            # Check if CUDA is available
            logging.info(f"Training on cuda: {torch.cuda.is_available()} -- {torch.cuda.device_count()} devices.")
            logging.info(f"Training on mps: {torch.mps.is_available()} -- {torch.mps.device_count()} devices.")

            config = ALL_CONFIGS["equiv_unet"].model

            # Create the model
            self.model = create_model(config).to(device)
            self.x = torch.ones((16, 1, 128, 128, 10), device=device)

        def test_forward(
            self,
        ):
    
            # Forward pass
            y = self.model(self.x)

            # Check the output shape
            self.assertEqual(y.shape[2:], self.x.shape[2:])

        def test_backward(
            self,
        ):
    
            # Forward pass
            y = self.model(self.x)

            # Loss
            loss = y.sum()

            # Backward pass
            loss.backward()

        def test_timing(
            self,
        ):
    
            # Forward pass
            for i in range(10):
                t0 = time.perf_counter()
                y = self.model(self.x)
                t1 = time.perf_counter()
                loss = y.sum()
                loss.backward()
                t2 = time.perf_counter()

                logging.info(f"Step {i} -- Forward: {t1-t0:.2f}s -- Backward: {t2-t1:.2f}s")

if __name__ == "__main__":
    absltest.main()