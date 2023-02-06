# User specific aliases and functions
alias sing_scqm_original='singularity shell --writable -B /cluster/dataset/medinfmk/scqm:/opt /cluster/dataset/medinfmk/scqm/containers/sing_scqm_611d975.simg'
alias sing_scqm='singularity shell --writable -B /cluster/dataset/medinfmk/scqm:/opt /cluster/dataset/medinfmk/scqm/containers/scqm.img'
alias sing_scqm2='singularity shell -H $HOME -B /cluster/dataset/medinfmk/scqm:/opt /cluster/dataset/medinfmk/scqm/containers/scqm.img'
alias sing_cecile_gpu='srun --cpus-per-task 2 --mem-per-cpu 13G --time 48:00:00 -p gpu --gpus-per-node=1 --pty singularity shell --nv -H $HOME -B /cluster/work/medinfmk/scqm:/opt /cluster/work/medinfmk/scqm/containers/scqm.img'
# jupyter notebook
alias jup_cecile='jupyter notebook --no-browser --ip=$(hostname -i) --port 6070'
# gpu job
alias gpu_job='srun --job-name gpu_inter --cpus-per-task 3 --mem-per-cpu 15G --time  72:00:00 -p gpu --gres=gpu:rtx3090:1 --pty bash'





# Path
export PYTHONPATH=$PYTHONPATH:/cluster/work/medinfmk/scqm/code/scqm_ct/scqm

# Custom package install directory
export PYTHONPATH=$PYTHONPATH:/opt/code/install_dir/lib/python3.9/site-packages

# venv
export PATH="$HOME/.local/bin:$PATH"
