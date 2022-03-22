# User specific aliases and functions
alias sing_scqm_original='singularity shell --writable -B /cluster/dataset/medinfmk/scqm:/opt /cluster/dataset/medinfmk/scqm/containers/sing_scqm_611d975.simg'
alias sing_scqm='singularity shell --writable -B /cluster/dataset/medinfmk/scqm:/opt /cluster/dataset/medinfmk/scqm/containers/scqm.img'
alias sing_scqm2='singularity shell -H $HOME -B /cluster/dataset/medinfmk/scqm:/opt /cluster/dataset/medinfmk/scqm/containers/scqm.img'
alias sing_cecile='bsub -W 10:00 -n 2 -Is singularity shell -H $HOME -B /cluster/dataset/medinfmk/scqm:/opt /cluster/dataset/medinfmk/scqm/containers/scqm.img'
alias sing_cecile_gpu='bsub -W 23:00 -n 2 -R "rusage[mem=4500,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" -Is singularity shell --nv -H $HOME -B /cluster/dataset/medinfmk/scqm:/opt /cluster/dataset/medinfmk/scqm/containers/scqm.img'
# Creates two different entries for running Jupyter notebooks
alias jup_login='jupyter notebook --no-browser --ip=127.0.0.1 --port 8787'
alias jup_batch='jupyter notebook --no-browser --ip=$(hostname -i) --port 8788'
alias jup_cecile='jupyter notebook --no-browser --ip=$(hostname -i) --port 6070'
# Create batch-job
alias bjob='bsub -Is -W 8:00 -n 2 -R "rusage[mem=4096]" bash'

# Path
export PYTHONPATH=$PYTHONPATH:/cluster/dataset/medinfmk/scqm/code/scqm_ct/scqm

# Custom package install directory
export PYTHONPATH=$PYTHONPATH:/opt/code/install_dir/lib/python3.9/site-packages
