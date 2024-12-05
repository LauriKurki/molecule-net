"""Root directories for datasets."""

from typing import Optional
import os


def get_root_dir(dataset) -> Optional[str]:
    """Get the root directory for the dataset."""
    hostname, username = os.uname()[1], os.environ.get("USER")

    if 'triton' in hostname:
        return f'/scratch/phys/project/sin/lauri/data/{dataset}/'
    elif 'd22' in hostname:
        return f"/l/data/molnet/{dataset}/"
    elif 'GHL96JPW91' in hostname:
        return f"/Users/kurkil1/data/{dataset}/"
    else:
        return f'/scratch/project_2005247/lauri/data/{dataset}/'
    return None
