#!/bin/bash

#SBATCH --job-name=train-test
#SBATCH --time=0:10:00
#SBATCH --output=logs/%x.out
#SBATCH --partition=gpu-debug
#SBATCH --gpus=1
#SBATCH --cpus-per-task 8
#SBATCH --mem-per-cpu 4G


# Load the required modules
source /scratch/work/kurkil1/venvs/molnet-torch/bin/activate

# Print environment 
pip list

# Run the model test
srun python -m torch_tests.train_test
