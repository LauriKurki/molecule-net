import ml_collections

def get_config() -> ml_collections.ConfigDict:
    """Get the default configuration."""
    config = ml_collections.ConfigDict()
    
    # Dataset
    config.dataset = "edafm"
    config.rng_seed = 0
    config.max_atoms = 54
    config.train_molecules = (0, 10000)
    config.val_molecules = (10000, 11000)
    config.test_molecules = (11000, 12000)

    config.shuffle_datasets = True
    config.batch_size = 32

    return config