import os
import re

from absl import logging

import jax
import flax
import tensorflow as tf

import chex
import ml_collections

from molnet.data import augmentation, utils

from typing import Dict, List, Sequence, Optional


def get_datasets(
    config: ml_collections.ConfigDict,
) -> Dict[str, tf.data.Dataset]:
    """Loads datasets for each split."""

    filenames = sorted(os.listdir(config.root_dir))
    filenames = [
        os.path.join(config.root_dir, f)
        for f in filenames
        if f.startswith("afms_")
    ]

    if len(filenames) == 0:
        raise ValueError(f"No files found in {config.root_dir}.")
    
    # Partition the filenames into train, val, and test.
    def filter_by_molecule_number(
        filenames: Sequence[str], start: int, end: int
    ) -> List[str]:
        def filter_file(filename: str, start: int, end: int) -> bool:
            filename = os.path.basename(filename)
            file_start, file_end = [int(val) for val in re.findall(r"\d+", filename)]
            return start <= file_start and file_end <= end

        return [f for f in filenames if filter_file(f, start, end)]

    # Number of molecules for training can be smaller than the chunk size.
    files_by_split = {
        "train": filter_by_molecule_number(filenames, *config.train_molecules),
        "val": filter_by_molecule_number(filenames, *config.val_molecules),
    }

    element_spec = tf.data.Dataset.load(filenames[0]).element_spec
    datasets = {}
    for split, files_split in files_by_split.items():

        dataset_split = tf.data.Dataset.from_tensor_slices(files_split)
        dataset_split = dataset_split.interleave(
            lambda path: tf.data.Dataset.load(path, element_spec=element_spec),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )

        # Shuffle the dataset.
        if split == 'train':
            dataset_split = dataset_split.shuffle(1000, seed=config.rng_seed, reshuffle_each_iteration=True)

        # Repeat the dataset.
        dataset_split = dataset_split.repeat()

        # batches consist of a dict {'x': image, 'xyz': xyz, 'sw': scan window})
        dataset_split = dataset_split.map(
            lambda x: {
                "images": x["x"],
                "xyz": x["xyz"],
                "sw": x["sw"],
            },
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )

        # Preprocess images.
        dataset_split = dataset_split.map(
            lambda x: _preprocess_images(
                x,
                config.noise_std,
                interpolate_z=config.interpolate_input_z,
                cutout_probs=config.cutout_probs,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )

        dataset_split = dataset_split.map(
            lambda x: _compute_atom_maps(
                x,
                z_cutoff=config.z_cutoff,
                sigma=config.sigma,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False
        )

        # Batch the dataset.
        dataset_split = dataset_split.batch(config.batch_size)
        dataset_split = dataset_split.prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
        
        datasets[split] = dataset_split
    return datasets


def _preprocess_images(
    batch: Dict[str, tf.Tensor],
    noise_std: float = 0.0,
    interpolate_z: Optional[int] = None,
    cutout_probs: Optional[List[float]] = [0.5, 0.3, 0.1, 0.05, 0.05],
) -> Dict[str, tf.Tensor]:
    """Preprocesses images."""
    
    x = batch["images"]
    x = tf.transpose(x, perm=[1, 0, 2])
    sw = batch["sw"]
    xyz = batch["xyz"]

    # Shift xyz coordinates by scan window, so that scan window starts at (0, 0).
    shifted_xyz = xyz[:, :2] - sw[0, :2]
    shifted_xyz = tf.concat([shifted_xyz, xyz[:, 2:]], axis=-1)

    # Also shift the scan window to start at (0, 0).
    shifted_sw = sw - sw[0]

    # Normalize the images to zero mean and unit variance.
    x = augmentation.normalize_images(x)

    # Add channel dimension.
    x = x[..., tf.newaxis]

    # Interpolate to `interpolate_z` z slices
    if interpolate_z is not None:
        x = tf.image.resize(x, (x.shape[1], interpolate_z), method='bilinear')

    # Add noise to the images.
    if noise_std > 0.0:
        x = x + tf.random.normal(tf.shape(x), stddev=noise_std)

    # Apply rotation and flip augmentation.
    #x, xyz = augmentation.random_rotate_3d_stacks_with_coords(x, xyz)
    #x, xyz = augmentation.random_flip_3d_stacks_with_coords(x, xyz, x.shape[:-1])

    # Create cutout augmentation.
    x = augmentation.add_random_cutouts(x, cutout_probs=cutout_probs, cutout_size_range=(5, 10))

    sample = {
        "images": x,
        "xyz": shifted_xyz,
        "sw": shifted_sw,
    }
    
    return sample


def _compute_atom_maps(
    batch: Dict[str, tf.Tensor],
    z_cutoff: float = 1.0,
    sigma: float = 0.2,
) -> tf.Tensor:
    """Computes atom maps."""
    xyz = batch["xyz"]
    z_max = tf.reduce_max(xyz[:, 2])

    # Compute grids # TODO: REPLACE WITH COORDINATES FROM BATCH["sw"]
    # Tried, didn't work. Come back to this later.
    # For now, the molecule (and scan window) is shifted in _preprocess_images to start at (0, 0).
    x = tf.linspace(0., 16., 128)
    y = tf.linspace(0., 16., 128)
    z = tf.linspace(z_max-z_cutoff, z_max, 10)
    X, Y, Z = tf.meshgrid(x, y, z, indexing='xy')

    # Compute atom maps.
    maps_h = tf.zeros_like(X)
    maps_c = tf.zeros_like(X)
    maps_n = tf.zeros_like(X)
    maps_o = tf.zeros_like(X)
    maps_f = tf.zeros_like(X)

    for atom in xyz:
        if atom[-1] == 1:
            maps_h += tf.exp(
                -((X - atom[0])**2 + (Y - atom[1])**2 + (Z - atom[2])**2) / (2 * sigma**2)
            )
        elif atom[-1] == 6:
            maps_c += tf.exp(
                -((X - atom[0])**2 + (Y - atom[1])**2 + (Z - atom[2])**2) / (2 * sigma**2)
            )
        elif atom[-1] == 7:
            maps_n += tf.exp(
                -((X - atom[0])**2 + (Y - atom[1])**2 + (Z - atom[2])**2) / (2 * sigma**2)
            )
        elif atom[-1] == 8:
            maps_o += tf.exp(
                -((X - atom[0])**2 + (Y - atom[1])**2 + (Z - atom[2])**2) / (2 * sigma**2)
            )
        elif atom[-1] == 9:
            maps_f += tf.exp(
                -((X - atom[0])**2 + (Y - atom[1])**2 + (Z - atom[2])**2) / (2 * sigma**2)
            )

    atom_map = tf.stack([maps_h, maps_c, maps_n, maps_o, maps_f], axis=0)
    atom_map = tf.transpose(atom_map, perm=[1, 2, 3, 0])

    return {
        "images": batch["images"],
        "xyz": batch["xyz"],
        "sw": batch["sw"],
        "atom_map": atom_map,
    }
