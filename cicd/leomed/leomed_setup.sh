# User specific aliases and functions
alias sing_scqm_original='singularity shell --writable -B /cluster/dataset/medinfmk/scqm:/opt /cluster/dataset/medinfmk/scqm/containers/sing_scqm_611d975.simg'
alias sing_scqm='singularity shell --writable -B /cluster/dataset/medinfmk/scqm:/opt /cluster/dataset/medinfmk/scqm/containers/scqm.img'
alias sing_scqm2='singularity shell -H $HOME -B /cluster/dataset/medinfmk/scqm:/opt /cluster/dataset/medinfmk/scqm/containers/scqm.img'
alias sing_cecile_gpu='srun --cpus-per-task 4 --mem-per-cpu 13G --time 24:00:00 -p gpu --gpus-per-node=1 --pty singularity shell --nv -H $HOME -B /cluster/work/medinfmk/scqm:/opt /cluster/work/medinfmk/scqm/containers/scqm.img'
# Creates two different entries for running Jupyter notebooks
alias jup_cecile='jupyter notebook --no-browser --ip=$(hostname -i) --port 6070'





# Path
export PYTHONPATH=$PYTHONPATH:/cluster/work/medinfmk/scqm/code/scqm_ct/scqm

# Custom package install directory
export PYTHONPATH=$PYTHONPATH:/opt/code/install_dir/lib/python3.9/site-packages
