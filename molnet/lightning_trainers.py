import os

import jax
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import lightning as L

import ml_collections

from molnet import graphics
from molnet.torch_models import create_model


class LightningMolnet(L.LightningModule):
    def __init__(self, config: ml_collections.ConfigDict, workdir: str):
        super().__init__()
        self.config = config
        self.workdir = workdir
        self.model = create_model(config.model)
        self.save_hyperparameters(self.config.to_dict())

    def training_step(self, batch, batch_idx):
       # batch = self.transfer_batch_to_device(batch, self.device, batch_idx)

        # Extract the inputs and targets
        x, atom_map, xyz = batch

        # Forward pass
        pred = self.model(x)

        # Compute the loss
        z_slices = x.shape[-1]
        loss = F.mse_loss(pred, atom_map[..., -z_slices:])
        self.log('train_loss', loss, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        #batch = self.transfer_batch_to_device(batch, self.device, batch_idx)

        # Extract the inputs and targets
        x, atom_map, xyz = batch

        # Forward pass
        pred = self.model(x)

        # Compute the loss
        z_slices = x.shape[-1]
        loss = F.mse_loss(pred, atom_map[..., -z_slices:])
        self.log('val_loss', loss, logger=True)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        # Convert numpy arrays to torch tensors
        batch = self.transfer_batch_to_device(batch, self.device, batch_idx)

        # Extract the inputs and targets
        x, atom_map, xyz = batch
        z_slices = x.shape[-1]

        # Forward pass
        pred, attention_maps = self.model(x, return_attention_maps=True)

        # Compute the loss
        y = atom_map[..., -z_slices:]
        loss_by_image = torch.mean((pred - y) ** 2, dim=(1, 2, 3, 4))

        # Transfer everything back to the CPU and numpy
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        pred = pred.detach().cpu().numpy()
        xyz = xyz.cpu().numpy()

        if self.config.batch_order == "torch":
            x = np.transpose(x, (0, 2, 3, 4, 1))
            y = np.transpose(y, (0, 2, 3, 4, 1))
            pred = np.transpose(pred, (0, 2, 3, 4, 1))
            attention_maps = [a.detach().cpu().numpy() for a in attention_maps]

        # Save predictions
        graphics.save_predictions(
            x, y, pred, loss_by_image, self.output_dir, batch_idx*self.config.batch_size
        )

        graphics.save_predictions_as_molecules(
            x, y, pred, xyz, self.output_dir,
            peak_threshold=self.config.peak_threshold,
            start_save_idx=batch_idx*self.config.batch_size
        )

        graphics.save_attention_maps(
            x, attention_maps, self.output_dir, batch_idx*self.config.batch_size
        )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # get current step
        step = self.global_step

        # Predict, if required
        if step % self.config.predict_every_steps == 0:
            for i in range(self.config.predict_num_batches):
                # get batch from validation dataloader
                batch = next(self.trainer.val_dataloaders)
                self.predict_step(batch, i, 0)

        # create output directory
        self.output_dir = os.path.join(
            self.workdir,
            "predictions",
            f"step_{step}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
