import os
import ml_collections

import torch
import lightning
from lightning.pytorch import loggers, callbacks

from molnet.data import input_pipeline_wds
from molnet.lightning_trainers import LightningMolnet


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Set up writer for logs
    tensorboard_logger = loggers.TensorBoardLogger(
        save_dir=workdir,
        name="lightning_logs",
    )

    # Set up checkpointing
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="ckpt-{step:06d}-{val_loss:.2e}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    trainer_callbacks = [
        checkpoint_callback
    ]

    # Set up trainer module
    trainer = lightning.Trainer(
        accelerator="gpu",
        default_root_dir=workdir,
        logger=tensorboard_logger,
        callbacks=trainer_callbacks,
        max_steps=config.num_train_steps,
        log_every_n_steps=config.log_every_steps,
        limit_val_batches=config.num_eval_steps,
        val_check_interval=config.eval_every_steps,
        
    )

    # Set up model
    model = LightningMolnet(config, workdir)

    # Set up data loaders
    ds = input_pipeline_wds.get_datasets(
        config=config,
    )
    train_loader = ds["train"]
    val_loader = ds["val"]

    # Train the model
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
