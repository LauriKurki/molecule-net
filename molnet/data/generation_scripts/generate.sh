#!/bin/bash

python wds_atom_map_generator.py \
    --data_dir /l/data/small_fragments/afm.h5 \
    --output_dir /l/data/molnet/atom_maps/ \
    --chunk_size 256 \
    --z_cutoff 2.5 \
    --n_molecules 2048 \
    --num_workers 4 \
