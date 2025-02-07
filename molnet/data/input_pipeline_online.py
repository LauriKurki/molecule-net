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
                sigma=config.sigma,
                factor=config.gaussian_factor,
                dataset=config.dataset 
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True
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
    
    x = batch["images"]
    x = tf.transpose(x, perm=[1, 0, 2])
    sw = batch["sw"]
    xyz = batch["xyz"]

    # Shift xyz coordinates by scan window, so that scan window starts at (0, 0).
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

    # Crop to size 128x128 after rotation to avoid artifacts.
    if x.shape[0] > 128:
        # Shift xyz by half of the crop size.
        shifted_xyz = shifted_xyz + (128 - x.shape[0]) * 0.125 / 2
        shifted_xyz = tf.concat([shifted_xyz[:, :2], xyz[:, 2:]], axis=-1)
        
        # Crop the images.
        x = augmentation.center_crop(x, 128)

    # Add noise to the images.
    x = augmentation.add_noise(x, noise_std)

    # Create cutouts.
    x = augmentation.add_random_cutouts(
        x, cutout_probs=cutout_probs, cutout_size_range=(2, 10), image_size=128
    )

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
    factor: float = 5.0,
    dataset: str = None,
) -> tf.Tensor:
    """Computes atom maps."""
    xyz = batch["xyz"]
    z_max = tf.reduce_max(xyz[:, 2])

    # Compute grids # TODO: REPLACE WITH COORDINATES FROM BATCH["sw"]
    # Tried, didn't work. Come back to this later.
    # For now, the molecule (and scan window) is shifted in _preprocess_images to start at (0, 0).
    x = tf.linspace(0., 16., 128)
    y = tf.linspace(0., 16., 128)
    z_steps = tf.cast(z_cutoff / 0.1, tf.int32)
    #z = tf.linspace(z_max-z_cutoff, z_max, z_steps)

    # Reversing the z axis seems to be essential for smooth learning.
    z = tf.linspace(z_max, z_max-z_cutoff, z_steps)

    X, Y, Z = tf.meshgrid(x, y, z, indexing='xy')

    # Compute atom maps.
    maps_h = tf.zeros_like(X)
    maps_c = tf.zeros_like(X)
    maps_n = tf.zeros_like(X)
    maps_o = tf.zeros_like(X)
    maps_f = tf.zeros_like(X)
    maps_cl = tf.zeros_like(X)
    maps_br = tf.zeros_like(X)

    for atom in xyz:
        m = tf.exp(
            -((X - atom[0])**2 + (Y - atom[1])**2 + (Z - atom[2])**2) / (2 * sigma**2)
        )
        
        # all values below 1e-4 to 0
        m = tf.where(m < 1e-2, tf.zeros_like(m), m*factor)

        if atom[-1] == 1:
            maps_h += m
        elif atom[-1] == 6:
            maps_c += m
        elif atom[-1] == 7:
            maps_n += m
        elif atom[-1] == 8:
            maps_o += m
        elif atom[-1] == 9:
            maps_f += m
        elif atom[-1] == 17:
            maps_cl += m
        elif atom[-1] == 35:
            maps_br += m

    if "bromine" in dataset:
        #atom_map = tf.stack([maps_h, maps_c, maps_n, maps_o, maps_f, maps_si, maps_p, maps_s, maps_cl, maps_br], axis=0)
        atom_map = tf.stack([maps_h, maps_c, maps_n, maps_o, maps_f, maps_cl, maps_br], axis=0)
    elif "water" in dataset:
        atom_map = tf.stack([maps_h, maps_o], axis=0)
    else:
        atom_map = tf.stack([maps_h, maps_c, maps_n, maps_o, maps_f], axis=0)

    atom_map = tf.transpose(atom_map, perm=[1, 2, 3, 0])

    return {
        "images": batch["images"],
        "xyz": batch["xyz"],
        "sw": batch["sw"],
        "atom_map": atom_map,
    }


def get_full_molecule_datasets(
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
        #dataset_split = dataset_split.repeat()

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

        # Only include molecules where all atoms are within target_z_cutoff of max z.
        # i.e. filter out molecules with any(z < z_max - target_z_cutoff - sigma)
        dataset_split = dataset_split.filter(
            lambda x: tf.reduce_all(
                (
                    x["xyz"][:, 2] > 
                    tf.reduce_max(x["xyz"][:, 2]) - config.target_z_cutoff - config.sigma
                )
            ),
        )

        # Preprocess
        dataset_split = dataset_split.map(
            lambda x: _preprocess_images(
                x,
                noise_std=config.noise_std,
                interpolate_z=config.interpolate_input_z,
                z_cutoff=config.z_cutoff,
                cutout_probs=config.cutout_probs,
                max_shift_per_slice=config.max_shift_per_slice,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )

        # Compute atom maps.
        dataset_split = dataset_split.map(
            lambda x: _compute_atom_maps(
                x,
                z_cutoff=config.target_z_cutoff,
                sigma=config.sigma,
                factor=config.gaussian_factor,
                include_heavy_atoms="bromine" in config.dataset 
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True
        )

        # Batch the dataset.
        dataset_split = dataset_split.batch(config.batch_size)
        dataset_split = dataset_split.prefetch(tf.data.AUTOTUNE).as_numpy_iterator()

        datasets[split] = dataset_split
    return datasets