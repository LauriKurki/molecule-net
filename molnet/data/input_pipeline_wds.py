import os
import re

import numpy as np
import webdataset as wds
import ml_collections

from typing import List, Tuple, Sequence, Dict


def get_datasets(config: ml_collections.ConfigDict):
    filenames = sorted(os.listdir(config.root_dir))
    filenames = [
        os.path.join(config.root_dir, f)
        for f in filenames
        if f.startswith("maps_")
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

    datasets = {}
    for split, files_split in files_by_split.items():
        ds = wds.WebDataset(
            files_split, resampled=True, shardshuffle=True
        )

        if config.shuffle_datasets:
            ds = ds.shuffle(1000, seed=config.rng_seed)

        # Decode bytes objects to torch tensors.
        ds = ds.decode("torch")

        # Apply image preprocessing and create sample (tuple)
        ds = ds.map(
            lambda x: make_sample(
                x,
                noise_std=config.noise_std,
                max_atoms=config.max_atoms
            )
        )

        ds = ds.batched(config.batch_size)

        loader = wds.WebLoader(
            ds, batch_size=None, num_workers=config.num_workers
        )

        # Unbatch, shuffle between workers, and batch again. Quite slow.
        #loader = loader.unbatched().shuffle(1000).batched(config.batch_size)

        # We are using resampling for shards so the dataset is infinite.
        # We set an artificial epoch size.
        loader = loader.with_epoch(
            len(files_split) * config.chunk_size // config.batch_size
        )

        datasets[split] = iter(loader)

    return datasets


def make_sample(
    sample: Dict[str, np.ndarray],
    noise_std: float,
    max_atoms: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = sample["x.npy"]
    xyz = sample["xyz.npy"]
    atom_map = sample["map.npy"]

    # cast to float32
    x = x.astype(np.float32)
    xyz = xyz.astype(np.float32)
    atom_map = atom_map.astype(np.float32)

    # Normalize x to 0 mean, 1 std.
    xmean = np.mean(x, axis=(1, 2), keepdims=True)
    xstd = np.std(x, axis=(1, 2), keepdims=True)
    x = (x - xmean) / (xstd + 1e-9)

    # Add noise to images
    if noise_std > 0:
        x = x + np.random.uniform(-1, 1, size=x.shape).astype(x.dtype) * noise_std
    
    # Add channel dimension
    x = x[None, ...]

    # Pad xyz to max_atoms
    xyz = np.pad(xyz, [[0, max_atoms - xyz.shape[0]], [0, 0]])

    # get z slices from x and reshape atom_map
    z_slices = x.shape[-1]
    atom_map = atom_map[..., -z_slices:]

    return x, atom_map, xyz
