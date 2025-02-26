import sys
import os
import io
import re
import tqdm
import tqdm.contrib.concurrent

import tensorflow as tf
import numpy as np
import webdataset as wds

from typing import Any, Dict, Tuple, List


def save_afms(
    files: List[str],
    split: str,
    save_dir: str,
):
    signature = {
        "x": tf.TensorSpec(shape=(160, 160, 15), dtype=tf.uint8),
        "xyz": tf.TensorSpec(shape=(None, 5), dtype=tf.float32),
        "sw": tf.TensorSpec(shape=(2, 3), dtype=tf.float64),
    }
    start_idx = int(files[0].split("_")[-1].split(".")[0])
    end_idx = int(files[-1].split("_")[-1].split(".")[0])
    save_name = os.path.join(
        save_dir,        
        f"{split}_afms_{start_idx}_{end_idx}"
    )

    def generator():
        for f in files:
            sample = np.load(f)
            #print(sample.keys())
            # x shape [Z, X, Y] -> [Y, X, Z]
            #print(f"x.shape: {x.shape}")

            yield {
                "x": sample["arr_0"].transpose(1, 2, 0),
                "xyz": sample["arr_1"],
                "sw": sample["arr_2"],
            }

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=signature
    )

    os.makedirs(save_name, exist_ok=True)
    ds.save(save_name)


def decode_xyz(key: str, data: Any):
    """
    Webdataset pipeline function for decoding xyz files.

    Arguments:
        key: Stream value key. If the key is ``'.xyz'``, then the data is decoded.
        data: Data to decode.

    Returns:
        Tuple (**xyz**, **scan_window**), where

        - **xyz** - Decoded atom coordinates and elements as an array where each row is of the form ``[x, y, z, element]``.
        - **scan_window** - The xyz coordinates of the opposite corners of the scan window in the form
          ``((x_start, y_start, z_start), (x_end, y_end, z_end))``

        If the stream key did not match, the tuple is ``(None, None)`` instead.
    """
    if key == ".xyz":
        data = io.BytesIO(data)
        atom_number = data.readline().decode("utf-8")
        comment = data.readline().decode("utf-8")
        sw = get_scan_window_from_comment(comment)
        xyz = []
        while line := data.readline().decode("utf-8"):
            e, x, y, z = line.strip().split()[:4]
            e = int(e)
            xyz.append([np.float32(x), np.float32(y), np.float32(z), e])
        return np.array(xyz).astype(np.float32), sw
    else:
        return None, None


def get_scan_window_from_comment(comment: str) -> np.ndarray:
    """
    Process the comment line in a .xyz file and extract the bounding box of the scan.
    The comment either has the format (QUAM dataset)

        ``Lattice="x0 x1 x2 y0 y1 y2 z0 z1 z2"``

    where the lattice is assumed to be orthogonal and origin at zero, or

        ``Scan window: [[x_start y_start z_start], [x_end y_end z_end]]``

    Arguments:
        comment: Comment to parse.

    Returns:
        The xyz coordinates of the opposite corners of the scan window in the form
            ``((x_start, y_start, z_start), (x_end, y_end, z_end))``
    """
    comment = comment.lower()
    match = re.match('.*lattice="((?:[+-]?(?:[0-9]*\.)?[0-9]+\s?){9})"', comment)
    if match:
        vectors = np.array([float(s) for s in match.group(1).split()])
        vectors = vectors.reshape((3, 3))
        sw = np.zeros((2, 3), dtype=np.float32)
        sw[1] = np.diag(vectors)
    elif match := re.match(
        r".*scan window: [\[(]{2}\s*((?:[+-]?(?:[0-9]*\.)?[0-9]+(?:e[-+]?[0-9]+)?,?\s*){3})[\])],\s*[\[(]\s*((?:[+-]?(?:[0-9]*\.)?[0-9]+(?:e[-+]?[0-9]+)?,?\s*){3})[\])]{2}.*",
        comment,
    ):
        start = np.array([float(s.strip(',')) for s in match.group(1).split()])
        end = np.array([float(s.strip(',')) for s in match.group(2).split()])
        sw = np.stack([start, end], axis=0)
    else:
        raise ValueError(f"Could not parse scan window in comment: `{comment}`")
    return sw


def batch_to_numpy(batch: Dict[str, Any]):
    images = []
    for k, v in batch.items():
        # If key ends in png, it is an image
        if k.endswith(".png"):
            images.append(v)
        # If key ends in .xyz, it is tuple (xyz, scan_window)
    xyz, sw = batch['xyz']

    # Keep only water hydrogen and oxygen atoms
    xyz = xyz[xyz[:, 3] < 9]

    # Add column to xyz for charge (index 3) value 0
    charge = np.zeros((xyz.shape[0], 1), dtype=np.float32)
    xyz = np.concatenate([xyz[:, :-1], charge, xyz[:, [-1]]], axis=1)

    x = np.stack(images, axis=0)
    return x, xyz, sw


def generator(dataloader):
    for sample in dataloader:
        yield batch_to_numpy(sample)
        
def _save_afm_wrapper(args):
    save_afms(*args)

if __name__=='__main__':
    local_scratch = sys.argv[1]
    # Read urls
    #directory = "/l/data/molnet/Water-bilayer"
    #directory = os.path.join(
    #    local_scratch,
    #    "SIN-AFM-FDBM"
    #)
    temp_dir = os.path.join(
        local_scratch,
        "SIN-AFM-FDBM-np"
    )
    save_dir = os.path.join(
        local_scratch,
        "SIN-AFM-FDBM-tf"
    )

    for split in ["train", "val", "test"]:
        #urls = [
        #    os.path.join(directory, f)
        #    for f in os.listdir(directory)
        #    if split in f
        #]

        # Create dataset
        #dataset = wds.WebDataset(urls).decode("pill", decode_xyz)
        #dl = iter(dataset)

        # First save the images to individual temporary files for later use
        #gen = generator(dl)
        #os.makedirs(temp_dir, exist_ok=True)

        #for i, batch in tqdm.tqdm(enumerate(gen)):
        #    np.savez(
        #        os.path.join(temp_dir, f"{split}_batch_{i:06}"),
        #        *batch
        #    )

        files = [
            os.path.join(temp_dir, f)
            for f in os.listdir(temp_dir)
            if f.endswith(".npz")
            and split in f
        ]

        files = sorted(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))

        chunk_size = 1024

        # Repeat files so that len(files) is a multiple of chunk_size
        n = len(files)

        r = n % chunk_size
        m = chunk_size - r
        if r != 0:
            files = files + files[:m]
        
        print(f"Saving {len(files)} files in chunks of {chunk_size} for split {split}")

        # Divide files into chunks of size chunk_size
        chunks = [
            files[i:i + chunk_size]
            for i in range(0, len(files), chunk_size)
        ]

        print(f"Saving {len(chunks)} chunks")
        print(f"First chunk: {chunks[0]}")

        # Save chunks (in parallel)
        args_list = [
            (
                chunk,
                split,
                save_dir
            ) for chunk in chunks
        ]
        tqdm.contrib.concurrent.process_map(
            _save_afm_wrapper,
            args_list,
            max_workers=20
        )

        # Save chunks (in serial)
        #for chunk in tqdm.tqdm(chunks):
        #    first_index = chunk[0].split("_")[-1].split(".")[0]
        #    last_index = chunk[-1].split("_")[-1].split(".")[0]
        #    save_name = f"{split}_afms_{first_index}_{last_index}"
        #    save_afms(
        #        chunk,
        #        os.path.join(save_dir, save_name)
        #    )


    # Remove temporary files in outputdir
    #for f in files:
    #    os.remove(f)

    print("Done")
