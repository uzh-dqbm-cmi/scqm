#!/bin/bash

#SBATCH --job-name=cv
#SBATCH --partition=gpu
#SBATCH --time=120:00:00


### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=1
#SBATCH --ntasks=5

#SBATCH --cpus-per-task=3


# Memory per node (In total, mem * nodes GB)
# SBATCH --mem=100g

#SBATCH --mem-per-cpu=12000

# Number of GPUs per node
#SBATCH --gres=gpu:5

#SBATCH -o /cluster/work/medinfmk/scqm/logs/out.out
# Path

export PYTHONPATH=$PYTHONPATH:/cluster/work/medinfmk/scqm/code/scqm_ct/scqm

srun -u --ntasks 1 --exclusive --gres=gpu:1 --gpus-per-task=1 --cpus-per-task=3 --mem-per-cpu=12000 python3 scqm/custom_library/trial_scripts/parallel_cv.py 0 &
srun -u --ntasks 1 --exclusive --gres=gpu:1 --gpus-per-task=1 --cpus-per-task=3 --mem-per-cpu=12000 python3 scqm/custom_library/trial_scripts/parallel_cv.py 1 &
srun -u --ntasks 1 --exclusive --gres=gpu:1 --gpus-per-task=1 --cpus-per-task=3 --mem-per-cpu=12000 python3 scqm/custom_library/trial_scripts/parallel_cv.py 2 &
srun -u --ntasks 1 --exclusive --gres=gpu:1 --gpus-per-task=1 --cpus-per-task=3 --mem-per-cpu=12000 python3 scqm/custom_library/trial_scripts/parallel_cv.py 3 &
srun -u --ntasks 1 --exclusive --gres=gpu:1 --gpus-per-task=1 --cpus-per-task=3 --mem-per-cpu=12000 python3 scqm/custom_library/trial_scripts/parallel_cv.py 4 &
wait
echo "All jobs completed"