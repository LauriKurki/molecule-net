#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --output=log_gen.out
#SBATCH -c 24
#SBATCH --mem-per-cpu=4G


source /scratch/work/kurkil1/venvs/molnet/bin/activate

python atom_map_generator.py \
    --data_dir /scratch/phys/project/sin/lauri/data/afm.h5 \
    --output_dir /scratch/phys/project/sin/lauri/data/atom_maps/ \
    --chunk_size 1000
