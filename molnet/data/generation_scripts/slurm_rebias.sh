#!/bin/bash
#SBATCH --account=project_2005247
#SBATCH --job-name=rebias_afm
#SBATCH --time=00:05:00
#SBATCH --output=log_rebias.out
#SBATCH -p small
#SBATCH --gres=nvme:1
#SBATCH -c 20
#SBATCH --mem-per-cpu=4G

# Load tensorflow/2.13
module load tensorflow/2.13
source /scratch/project_2005247/lauri/venvs/tf-2.13/bin/activate

# Copy mol_database to $LOCAL_SCRATCH
echo "Copying data to " $LOCAL_SCRATCH
tar -xvf /scratch/project_2005247/lauri/data/mol_database.tar.gz -C $LOCAL_SCRATCH
ls $LOCAL_SCRATCH

srun python rebias_xyzs.py \
    --database_path $LOCAL_SCRATCH/mol_database/ \
    --save_path /scratch/project_2005247/lauri/data/rotations_HCNOF_tol_0.3.pickle \
    --num_workers 20 \
    --flat_dist_tol 0.3
