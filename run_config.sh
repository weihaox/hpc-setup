# module avail cuda
# Each time you log in, it is necessary to load modules first. 
# Otherwise, you will encounter an error if directly typing python in the terminal:
# $ [<CRSid>@login-q-2 ~]$ python
# $ bash: python: command not found
# 
# virtualenv --system-site-packages ~/env-pytorch1.12
# source ~/env-pytorch1.12/bin/activate
# 
module load python/3.8
# module load miniconda/3
module load cuda/11.2
module load cudnn/8.1_cuda-11.2
source /home/<CRSid>/env-pytorch1.12/bin/activate