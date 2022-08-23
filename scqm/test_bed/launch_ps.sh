#!/bin/bash

#SBATCH --job-name=ps_app
#SBATCH --time=02:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=3

#SBATCH --mem-per-cpu=5G

#SBATCH -o /cluster/home/ctrottet/logs/ps.out

hostname -i

srun python3 -m RangeHTTPServer
