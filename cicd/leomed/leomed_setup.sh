# User specific aliases and functions
alias sing_scqm_original='singularity shell --writable -B /cluster/dataset/medinfmk/scqm:/opt /cluster/dataset/medinfmk/scqm/containers/sing_scqm_611d975.simg'
alias sing_scqm='singularity shell --writable -B /cluster/dataset/medinfmk/scqm:/opt /cluster/dataset/medinfmk/scqm/containers/scqm.img'
alias sing_scqm2='singularity shell -H $HOME -B /cluster/dataset/medinfmk/scqm:/opt /cluster/dataset/medinfmk/scqm/containers/scqm.img'

# Creates two different entries for running Jupyter notebooks
alias jup_login='jupyter notebook --no-browser --ip=127.0.0.1 --port 8787'
alias jup_batch='jupyter notebook --no-browser --ip=$(hostname -i) --port 8788'

# Create batch-job
alias bjob='bsub -Is -W 8:00 -n 2 -R "rusage[mem=4096]" bash'

# Path
export PYTHONPATH=$PYTHONPATH:/cluster/dataset/medinfmk/scqm/code/scqm/scqm

# Custom package install directory
export PYTHONPATH=$PYTHONPATH:/opt/code/install_dir/lib/python3.9/site-packages