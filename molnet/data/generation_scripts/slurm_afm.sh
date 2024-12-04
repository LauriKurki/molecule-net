#!/bin/bash
#SBATCH --account=project_2005247
#SBATCH --job-name=generate-maps
#SBATCH --time=01:00:00
#SBATCH --output=log_gen.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1,nvme:1
#SBATCH -c 2
#SBATCH --mem-per-cpu=4G

# Load tensorflow/2.13
module load tensorflow/2.13
source /scratch/project_2005247/lauri/venvs/tf-2.13/bin/activate

# Copy mol_database to $LOCAL_SCRATCH
echo "Copying data to " $LOCAL_SCRATCH
ls $LOCAL_SCRATCH
cp /scratch/project_2005247/lauri/data/rotations_HCNOF_tol_0.3.pickle $LOCAL_SCRATCH
tar -xf /scratch/project_2005247/lauri/data/mol_database.tar.gz -C $LOCAL_SCRATCH
ls $LOCAL_SCRATCH
echo "Done, starting AFM simulation"

srun python afm_generator.py \
    --molecule_dir $LOCAL_SCRATCH/mol_database/ \
    --rotations_fname $LOCAL_SCRATCH/rotations_HCNOF_tol_0.3.pickle \
    --output_dir /scratch/project_2005247/lauri/data/afms_rebias/ \
    --chunk_size 1024 \
    --batch_size 128 \

