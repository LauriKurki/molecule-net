"""Main file for running the training pipeline."""
import os

#os.environ["XLA_FLAGS"] = (
    #"--xla_gpu_enable_triton_softmax_fusion=true "
    #"--xla_gpu_triton_gemm_any=true "
    #"--xla_gpu_enable_async_collectives=true "
    #"--xla_gpu_enable_latency_hiding_scheduler=true "
    #"--xla_gpu_enable_highest_priority_async_stream=true "
#)

from absl import app
from absl import flags
from absl import logging

from clu import platform
from ml_collections import config_flags

import jax
import tensorflow as tf

from molnet import train, train_segmentation
from configs import root_dirs

FLAGS = flags.FLAGS


flags.DEFINE_string('workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True,
)

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], 'GPU')

    # Set tf random seed
    tf.random.set_seed(0)
    tf.random.set_global_generator(tf.random.Generator.from_seed(0))

    logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())

    # Add a note so that we can tell which task is which JAX host.
    # (Depending on the platform task 0 is not guaranteed to be host 0)
    platform.work_unit().set_task_status(
        f'process_index: {jax.process_index()}, '
        f'process_count: {jax.process_count()}'
    )
    platform.work_unit().create_artifact(
        platform.ArtifactType.DIRECTORY, FLAGS.workdir, 'workdir'
    )

    config = FLAGS.config

    # if the model is a segmentation model, use segmentation training
    if config.loss_fn in ["cross_entropy", "focal_loss", "dice_loss"]:
        train_segmentation.train_and_evaluate(config, FLAGS.workdir)
    else:
        train.train_and_evaluate(config, FLAGS.workdir)
    
if __name__ == "__main__":
    flags.mark_flags_as_required(['config', 'workdir'])
    app.run(main)
