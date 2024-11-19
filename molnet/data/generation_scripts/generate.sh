#!/bin/bash

python wds_atom_map_generator.py \
    --data_dir ~/data/small_afm.h5 \
    --output_dir ~/data/atom_maps \
    --chunk_size 32 \
    --z_cutoff 2.5 \
