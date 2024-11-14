#!/bin/bash
#SBATCH --account=project_2005247
#SBATCH --job-name=generate-maps
#SBATCH --time=02:00:00
#SBATCH --output=log_gen.out
#SBATCH -c 40
#SBATCH --mem-per-cpu=4G

source /scratch/project_2005247/lauri/venvs/molnet/bin/activate

python wds_atom_map_generator.py \
    --data_dir /scratch/project_2005247/lauri/data/afm.h5 \
    --output_dir /scratch/project_2005247/lauri/data/atom_maps/ \
    --chunk_size 1000
