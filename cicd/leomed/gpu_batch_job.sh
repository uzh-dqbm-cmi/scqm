#!/bin/bash

#SBATCH --job-name=scqm
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=6G


singularity shell --nv -H $HOME -B /cluster/work/medinfmk/scqm:/opt /cluster/work/medinfmk/scqm/containers/scqm.img
