import os
import re
import functools

import numpy as np
import scipy.ndimage
import webdataset as wds
import ml_collections

import torch.utils.data

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

    wds_cache_dir = './_cache'
    os.makedirs(wds_cache_dir, exist_ok=True)
    for split, files_split in files_by_split.items():

        ds = wds.DataPipeline(
            #wds.ResampledShards(files_split),
            wds.SimpleShardList(files_split),
            wds.shuffle(100),
                
            wds.split_by_worker,

            wds.tarfile_to_samples(),

            wds.shuffle(1000),

            wds.decode('l'),

            wds.map(
                functools.partial(
                    make_sample,
                    noise_std=config.noise_std,
                    max_atoms=config.max_atoms,
                    interpolate_z=config.interpolate_input_z
                )
            ),

            wds.shuffle(1000),
            wds.batched(config.batch_size),
        )

        #ds = torch.utils.data.DataLoader(
        ds = wds.WebLoader(
            ds,
            batch_size=None,
            shuffle=False,
            num_workers=0,
            #persistent_workers=True,
            #prefetch_factor=2,
        )

        ds = ds.repeat()

        # Unbatch, shuffle between workers, and batch again. Quite slow.
        #loader = loader.unbatched().shuffle(1000).batched(config.batch_size)

        # We are using resampling for shards so the dataset is infinite.
        # We set an artificial epoch size.
        #loader = loader.with_epoch(
        #    len(files_split) * config.chunk_size // config.batch_size
        #)

        datasets[split] = ds

    return datasets


def make_sample(
    sample: Dict[str, np.ndarray],
    noise_std: float,
    max_atoms: int,
    interpolate_z: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = sample["x.npy"]
    xyz = sample["xyz.npy"]
    atom_map = sample["map.npy"]

    # cast to float32
    x = x.astype(np.float32)
    xyz = xyz.astype(np.float32)
    atom_map = atom_map.astype(np.float32)
    
    # Normalize x to 0 mean, 1 std.
    xmean = np.mean(x, axis=(0, 1), keepdims=True)
    xstd = np.std(x, axis=(0, 1), keepdims=True)
    x = (x - xmean) / (xstd + 1e-9)

    # Interpolate z
    if interpolate_z is not None:
        zoom_factors = (1, 1, interpolate_z / x.shape[-1])      
        x = scipy.ndimage.zoom(x, zoom_factors)

    # Add noise to images
    if noise_std > 0:
        x = x + np.random.uniform(-1, 1, size=x.shape).astype(x.dtype) * noise_std
    
    # Add channel dimension
    x = x[..., None]

    # Pad xyz to max_atoms
    xyz = np.pad(xyz, [[0, max_atoms - xyz.shape[0]], [0, 0]])

    # get z slices from x and reshape atom_map
    z_slices = x.shape[-2]
    atom_map = atom_map[..., -z_slices:]

    # move first axis to last
    atom_map = np.transpose(atom_map, (1, 2, 3, 0))

    return x, atom_map, xyz
