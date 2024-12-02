"""Root directories for datasets."""

from typing import Optional
import os


def get_root_dir(local_scratch=None) -> Optional[str]:
    """Get the root directory for the dataset."""
    hostname, username = os.uname()[1], os.environ.get("USER")

    if local_scratch is not None:
        return local_scratch

    if 'triton' in hostname:
        return '/scratch/phys/project/sin/lauri/data/atom_maps/'
    elif 'd22' in hostname:
        return "/l/data/molnet/afms/"
    elif 'GHL96JPW91' in hostname:
        return "/Users/kurkil1/data/afms/"
    else:
        return '/scratch/project_2005247/lauri/data/afms/'
    return None
