export PYTHONPATH=$PYTHONPATH:/cluster/work/medinfmk/scqm/code/scqm_ct/scqm

# srun --ntasks 1 --cpus-per-task 2 --mem-per-cpu 5G --time 00:05 -u python3 scqm/custom_library/parameters/cv.py 5

sbatch scqm/cv.sh