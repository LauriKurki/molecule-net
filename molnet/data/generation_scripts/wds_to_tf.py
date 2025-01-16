import os
import io
import re
import tqdm

import numpy as np
import webdataset as wds

from typing import Any, Dict, Tuple


def decode_xyz(key: str, data: Any) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
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


if __name__=='__main__':
    # Read urls
    #directory = "/l/data/molnet/Water-bilayer"
    directory = "/scratch/project_2005247/lauri/data/Water-bilayer"
    outputdir = "/scratch/project_2005247/lauri/data/Water-bilayer-temp"
    urls = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
    ]

    # Create dataset
    dataset = wds.WebDataset(urls).decode("l", decode_xyz)
    dl = iter(dataset)

    # First save the images to individual temporary files for later use
    gen = generator(dl)
    os.makedirs(outputdir, exist_ok=True)

    for i, batch in tqdm.tqdm(enumerate(gen)):
        np.savez(
            os.path.join(outputdir, f"batch_{i}"),
            **batch
        )
