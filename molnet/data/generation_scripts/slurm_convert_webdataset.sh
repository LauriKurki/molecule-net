#!/bin/bash
#SBATCH --account=project_2005247
#SBATCH --job-name=convert_wds
#SBATCH --time=04:00:00
#SBATCH --output=log_convert.out
#SBATCH -p small
#SBATCH --gres=nvme:1500
#SBATCH -c 4
#SBATCH --mem-per-cpu=4G

# Load tensorflow/2.13
module load tensorflow/2.13
source /scratch/project_2005247/lauri/venvs/tf-2.13/bin/activate

# Copy necessary data to $LOCAL_SCRATCH
echo "Copying data to " $LOCAL_SCRATCH
cp -r /scratch/project_2005247/lauri/data/SIN-AFM-FDBM/ $LOCAL_SCRATCH
ls $LOCAL_SCRATCH

# Run the conversion script
echo "Running conversion script"
t0=$(date +%s)
srun python molnet/data/generation_scripts/wds_to_tf.py $LOCAL_SCRATCH
t1=$(date +%s)
echo "Conversion took" $((t1-t0)) "seconds"

# Copy the results back to the original directory
echo "Copying results back to /scratch/project_2005247/lauri/data/SIN-AFM-FDBM-tf/"
cp -r $LOCAL_SCRATCH/SIN-AFM-FDBM-tf/ /scratch/project_2005247/lauri/data/
