#!/bin/bash

#SBATCH --job-name=job
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=3

# SBATCH --mem=50g

#SBATCH --mem-per-cpu=12000

#SBATCH --gres=gpu:1

#SBATCH -o /cluster/work/medinfmk/scqm/logs/job.out 
#
export PYTHONPATH=$PYTHONPATH:/cluster/work/medinfmk/scqm/code/scqm_ct/scqm
export PATH="$HOME/.local/bin:$PATH"
source ~/envir/scqm/bin/activate

python3 -u scqm/custom_library/trial_scripts/create_cv.py




