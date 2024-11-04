"""Root directories for datasets."""

from typing import Optional
import os


def get_root_dir(local_scratch=None) -> Optional[str]:
    """Get the root directory for the dataset."""
    hostname, username = os.uname()[1], os.environ.get("USER")

    if local_scratch is not None:
        return local_scratch

    if username == "kurkil1":
        if 'triton' in hostname:
            return None
        else:
            return "/l/data/molnet/atom_maps/"
    if username == 'kurkilau':
        return None
    return None