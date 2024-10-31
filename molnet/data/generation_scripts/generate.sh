#!/bin/bash


python atom_map_generator.py \
    --data_dir ~/work/molnet/triton/afm.h5 \
    --output_dir ~/work/molnet/triton/atom_maps \
    --chunk_size 1000
