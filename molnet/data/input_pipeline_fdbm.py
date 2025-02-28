import os
import re

import tensorflow as tf
import ml_collections

from molnet.data import augmentation

from typing import Dict, List, Sequence, Optional


def get_datasets(
    config: ml_collections.ConfigDict,
) -> Dict[str, tf.data.Dataset]:
    """Loads datasets for each split."""

    filenames = sorted(os.listdir(config.root_dir))
    filenames = [
        os.path.join(config.root_dir, f)
        for f in filenames
        if f.startswith("train_afms") or f.startswith("val_afms")
    ]
    files_by_split = {
        "train": [
            os.path.join(config.root_dir, f)
            for f in filenames
            if "train_afms" in f
        ],
        "val": [
            os.path.join(config.root_dir, f)
            for f in filenames
            if "val_afms" in f
        ],
    }

    element_spec = tf.data.Dataset.load(filenames[0]).element_spec
    datasets = {}
    for split, files_split in files_by_split.items():

        # Load the files.
        dataset_split = tf.data.Dataset.from_tensor_slices(files_split)
        # Shuffle the files.
        dataset_split = dataset_split.shuffle(1000)
        dataset_split = dataset_split.interleave(
            lambda path: tf.data.Dataset.load(path, element_spec=element_spec),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )

        # Repeat the dataset.
        dataset_split = dataset_split.repeat()

        # Shuffle the dataset.
        if split == 'train':
            dataset_split = dataset_split.shuffle(
                1000,
                reshuffle_each_iteration=True,
            )

        # batches consist of a dict {'x': image, 'xyz': xyz, 'sw': scan window})
        dataset_split = dataset_split.map(
            lambda x: {
                "images": x["x"],
                "xyz": x["xyz"],
                "sw": x["sw"],
            },
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )

        # Preprocess images.
        dataset_split = dataset_split.map(
            lambda x: _preprocess_images(
                x,
                config.noise_std,
                interpolate_z=config.interpolate_input_z,
                z_cutoff=config.z_cutoff,
                cutout_probs=config.cutout_probs,
                max_shift_per_slice=config.max_shift_per_slice,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )

        dataset_split = dataset_split.map(
            lambda x: _compute_atom_maps(
                x,
                z_cutoff=config.target_z_cutoff,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True
        )

        # Pad the xyzs to the same length (54).
        dataset_split = dataset_split.map(
            lambda x: {
                "images": x["images"],
                "xyz": tf.pad(x["xyz"], [[0, 54 - tf.shape(x["xyz"])[0]], [0, 0]]),
                "sw": x["sw"],
                "atom_map": x["atom_map"],
            },
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
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
    z_cutoff: float = 1.0,
    cutout_probs: Optional[List[float]] = [0.5, 0.3, 0.1, 0.05, 0.05],
    max_shift_per_slice: float = 0.02,
) -> Dict[str, tf.Tensor]:
    """Preprocesses images."""
    
    # cast x to float32
    x = batch["images"]
    #x = tf.transpose(x, perm=[1, 0, 2])
    x = x[::-1, :, :]
    x = tf.cast(x, tf.float32)
    
    # cast sw to float32
    sw = batch["sw"]
    sw = tf.cast(sw, tf.float32)

    # Shift xyz coordinates by scan window, so that scan window starts at (0, 0).
    xyz = batch["xyz"]
    shifted_xyz = xyz[:, :2] - sw[0, :2]
    shifted_xyz = tf.concat([shifted_xyz, xyz[:, 2:]], axis=-1)

    # Also shift the scan window to start at (0, 0).
    shifted_sw = sw - sw[0]

    # Crop slices to z_cutoff.
    z_slices = z_cutoff / 0.1
    x = x[..., -int(z_slices):]

    # Normalize the images to zero mean and unit variance.
    x = augmentation.normalize_images(x)

    # Add channel dimension.
    x = x[..., tf.newaxis]

    # Interpolate to `interpolate_z` z slices
    if interpolate_z is not None:
        x = tf.image.resize(x, (x.shape[1], interpolate_z), method='bilinear')

    # Randomly shift the slices.
    x = augmentation.random_slice_shift(x, max_shift_per_slice=max_shift_per_slice)

    # Rotate inputs tensor and coordinates.
    (
        x,
        shifted_xyz
    ) = augmentation.random_rotate(x, shifted_xyz)

    # Add noise to the images.
    x = augmentation.add_noise(x, noise_std)

    # Create cutouts.
    x = augmentation.add_random_cutouts(x, cutout_probs=cutout_probs, cutout_size_range=(2, 10))

    sample = {
        "images": x,
        "xyz": shifted_xyz,
        "sw": shifted_sw,
    }
    
    return sample


def _compute_atom_maps(
    batch: Dict[str, tf.Tensor],
    z_cutoff: float = 1.0,
) -> tf.Tensor:
    """Computes atom maps."""
    xyz = batch["xyz"]
    z_max = tf.reduce_max(xyz[:, 2])
    sw = batch["sw"]
    xres = batch["images"].shape[0]

    x = tf.linspace(sw[0,0], sw[1,0], xres)
    y = tf.linspace(sw[0,1], sw[1,1], xres)
    z_steps = tf.cast(z_cutoff / 0.1, tf.int32)
    z = tf.linspace(z_max, z_max-z_cutoff, z_steps)

    X, Y, Z = tf.meshgrid(x, y, z, indexing='xy')

    # Compute atom maps.
    atom_map = tf.zeros_like(X)

    for atom in xyz:
        # Skip padding atoms
        if atom[-1] == 0:
            break

        m = (X - atom[0])**2 + (Y - atom[1])**2 + (Z - atom[2])**2
        m = tf.where(m < 0.2, atom[-1], 0)

        atom_map += m

    # Cast atom map to int.
    atom_map = tf.cast(atom_map, tf.int32)

    return {
        "images": batch["images"],
        "xyz": batch["xyz"],
        "sw": batch["sw"],
        "atom_map": atom_map,
    }

def transform_input_and_target(
    batch: Dict[str, tf.Tensor],
) -> Dict[str, tf.Tensor]:
    """Transforms the input and target."""
    x = batch["images"]
    y = batch["atom_map"]

    # Center crop
    x, y = augmentation.center_crop(
        x, y, size=128, shift=8
    )

    return {
        "images": x,
        "atom_map": y,
        "xyz": batch["xyz"],
        "sw": batch["sw"],
    }
