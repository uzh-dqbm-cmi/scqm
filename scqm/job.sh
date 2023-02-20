#!/bin/bash

#SBATCH --job-name=knn
#SBATCH --partition=gpu
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=3


#SBATCH --mem-per-cpu=20000

#SBATCH --gres=gpu:rtx1080ti:1
###SBATCH --gres=gpu:rtx3090:1
#SBATCH -o /cluster/work/medinfmk/scqm/logs/shap_das28.out 
#

###export PYTHONPATH=$PYTHONPATH:/cluster/work/medinfmk/scqm/code/scqm_ct/scqm

export PYTHONPATH=$PYTHONPATH:/opt/code/install_dir/lib/python3.8/site-packages
export PATH="$HOME/.local/bin:$PATH"
source ~/envir/scqm/bin/activate
export PYTHONPATH=$PYTHONPATH:/opt/code/install_dir/lib/python3.8/site-packages

python3 -u scqm/custom_library/trial_scripts/find_similarities.py




