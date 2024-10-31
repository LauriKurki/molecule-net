#!/bin/bash

python generate_atom_maps.py \
    --data_dir ~/work/molnet/triton/afm.h5 \
    --output_dir ~/work/molnet/triton/atom_maps \
    --chunk_size 100
