#!/bin/bash

#SBATCH --job-name=create_cv
#SBATCH --partition=gpu
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=3


#SBATCH --mem-per-cpu=12000

###SBATCH --gres=gpu:rtx1080ti:1
#SBATCH --gres=gpu:rtx3090:1
#SBATCH -o /cluster/work/medinfmk/scqm/logs/train_mlp_asdas.out 
#
export PYTHONPATH=$PYTHONPATH:/cluster/work/medinfmk/scqm/code/scqm_ct/scqm
export PATH="$HOME/.local/bin:$PATH"
source ~/envir/scqm/bin/activate

python3 -u scqm/custom_library/trial_scripts/train_mlp.py




