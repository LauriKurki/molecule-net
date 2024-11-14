"""Main file for running the training pipeline."""

from absl import app
from absl import flags
from absl import logging

#from clu import platform
from ml_collections import config_flags

#import jax
import torch
#import tensorflow as tf

#from molnet import train
from molnet import train_torch
from configs import root_dirs

FLAGS = flags.FLAGS

flags.DEFINE_string('code', None, 'Code to run the training with (torch or jax).')
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

    if FLAGS.code == 'torch':

        # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
        # it unavailable to PyTorch.
       #tf.config.experimental.set_visible_devices([], 'GPU')

        # Check if CUDA is available
        logging.info(f"Training on cuda: {torch.cuda.is_available()} -- {torch.cuda.device_count()} devices.")
        
        config = FLAGS.config
        config.root_dir = root_dirs.get_root_dir()

        train_torch.train_and_evaluate(config, FLAGS.workdir)

    elif FLAGS.code == 'jax':
        # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
        # it unavailable to JAX.
        tf.config.experimental.set_visible_devices([], 'GPU')

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
        config.root_dir = root_dirs.get_root_dir()

        train.train_and_evaluate(config, FLAGS.workdir)
    else:
        raise ValueError(f'Code {FLAGS.code} not recognized.')
    
if __name__ == "__main__":
    flags.mark_flags_as_required(['code', 'config', 'workdir'])
    app.run(main)