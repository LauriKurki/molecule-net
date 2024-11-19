#!/bin/bash
#SBATCH --job-name=generate-maps
#SBATCH --time=02:00:00
#SBATCH --output=log_gen.out
#SBATCH -c 128
###SBATCH --mem-per-cpu=4G

source /scratch/work/kurkil1/venvs/molnet-torch/bin/activate

pip list

srun python wds_atom_map_generator.py \
    --data_dir /scratch/phys/project/sin/lauri/data/afm.h5 \
    --output_dir /scratch/phys/project/sin/lauri/data/atom_maps/ \
    --chunk_size 512 \
    --z_cutoff 3.0 \
