# Getting Pytorch working on HPC

## 0. overview
- The [university](https://www.hpc.cam.ac.uk/applications-access-research-computing-services) provides a detailed guidance on how to use the [Research Computing Services HPC](https://docs.hpc.cam.ac.uk/hpc/). This is a simple guide designed specifically for deep learning researchers who are new to working with HPC GPU machines.
- There are two kinds of nodes (wilkes2: pascal; wilkes3: ampere). Wilkes3 (ampere) is recommended as it's more powerful (NVIDIA A100 GPUs).
    * `Wilkes2-GPU`: pascal; RHEL7; login-gpu.hpc.cam.ac.uk; login-e-1.hpc.cam.ac.uk, …, login-e-4.hpc.cam.ac.uk
    * `Wilkes3-GPU`: icelake CPU or ampere GPU nodes; RHEL8; login-icelake.hpc.cam.ac.uk;  login-q-1.hpc.cam.ac.uk, …, login-q-4.hpc.cam.ac.uk.
- There are two ways of file and job management (1. login-web.hpc.cam.ac.uk or 2. terminal)

most used:
- login: `ssh <CRSid>@login-q-4.hpc.cam.ac.uk`
- environment: `hpc_setup/run_config.sh`
- slurm_batch: `sbatch my_slurm_submit.wilkes3`
- check active jod: `squeue -u <CRSid>`

## 1. login 

See [Connecting to CSD3](https://docs.hpc.cam.ac.uk/hpc/user-guide/connecting.html).
 
- no vpn required;
- for the first time, you have to use the special node `ssh <CRSid>@multi.hpc.cam.ac.uk` to configure TOTP. This will give you a QR code and use google/microsoft authenticator to scan the code (or mannually enter the secret key).  
- then you will be allowed to use other HPC host: e.g. `ssh <CRSid>@login-q-4.hpc.cam.ac.uk` (RHEL8: login-q-1 to login-q-4). This will need UIS Password and TOTP Verification Code.

If you want to connect Visual Studio Code to the HPC, [here](https://cambiotraining.github.io/hpc-intro/02-working_on_hpc.html) gives a comprehensive instruction.

## 2. environment

See [Python](https://docs.hpc.cam.ac.uk/hpc/software-tools/python.html) and [PyTorch](https://docs.hpc.cam.ac.uk/hpc/software-packages/pytorch.html) or any other desired environments.

Each time you log in, it is necessary to load modules first. Otherwise, you will encounter an error if directly typing python in the terminal:
```Shell
$ [<CRSid>@login-q-2 ~]$ python
$ bash: python: command not found
```
It is recommended to store all of these modules in a `hpc_setup/run_config.sh` file or simply enter the commands in the terminal.  Using Anaconda and Virtual Environments (`virtualenv` or `conda` environments) are recommanded. You should config the environment first and load the same environment in the `slurm_submit` file. You can also use `requirements.txt` or `python_env.txt` to manage the python packages. Examples can be found [here](https://github.com/Aaron-Zhao123/gpu_cluster_setup) and [here](https://github.com/adianliusie/hpc_setup).

```Shell
# gpu_cluster_setup/run_config.sh

module load python/3.8
module load cuda/11.2
module load cudnn/8.1_cuda-11.2
source /home/<CRSid>/env-pytorch1.12/bin/activate
```
<!-- `slurm_submit.wilkes2`
To submit a job, simply run: `sbatch slurm_submit.wilkes2`.  -->

Below are some useful commands:
```Shell
module load <module>         -> load module
module unload <module>       -> unload module
module purge                 -> unload all modules
module list                  -> show currently loaded modules
module avail                 -> show available modules
module whatis                -> show available modules with brief explanation
```

Be careful with the versions. Please check [the version compatibility for pytorch](https://github.com/pytorch/pytorch/wiki/PyTorch-Versions) and [install](https://pytorch.org/get-started/previous-versions/) the required version. The command `module avail cuda` will give you the available CUDA modules.

## 3. file system (where to put the codes and datasets)

See [Data Transfer Guide](https://docs.hpc.cam.ac.uk/hpc/user-guide/transfer.html) and [File and I/O Management](https://docs.hpc.cam.ac.uk/hpc/user-guide/io_management.html).

There are two dictionaries for you to put the codes and datasets.
- `/home/<CRSid>/` (50GB);
- `/rds/user/<CRSid>/hpc-work` ((1TB and 1 million files) -> this is soft linked to `/home/<CRSid>/rds/hpc-work`.

It is suggested to store large datasets in `~/rds/hpc-work`, especially when it is large.

There are several ways to transfer your datasets to the target path: upload via `scp` or [Login-Web](https://login-web.hpc.cam.ac.uk); download via `gdown` from google drive or `wget` from somewhere else. 

[Login-Web](https://login-web.hpc.cam.ac.uk) (Research Computing Services HPC) can be used for both file and job management. Please see the [online documentation](https://docs.hpc.cam.ac.uk/hpc/user-guide/login-web.html) for further information.

Through this web portal, you can [upload and download files](https://docs.hpc.cam.ac.uk/hpc/user-guide/login-web.html#file-transfer-and-management) to and from CSD3, [create, edit, and submit SLURM jobs](https://docs.hpc.cam.ac.uk/hpc/user-guide/login-web.html#job-management), [run from a list of GUI applications](https://docs.hpc.cam.ac.uk/hpc/user-guide/login-web.html#interactive-apps), and [launch a full remote desktop](https://docs.hpc.cam.ac.uk/hpc/user-guide/login-web.html#remote-desktop) or [login node command line](https://docs.hpc.cam.ac.uk/hpc/user-guide/login-web.html#shell-access), all via a web browser, with no client software to install and configure on your local machine.

## 4. code

It is [advisable](https://docs.hpc.cam.ac.uk/hpc/user-guide/a100.html) to check if the code is runable and the system environment is appropriate before submitting the job to the queue. For example, you can check the code 1) on `CL GPUs` (e.g. dev-gpu-1.cl.cam.ac.uk); or 2) by requesting an [interactive node](https://docs.hpc.cam.ac.uk/hpc/user-guide/a100.html#software):
`sintr -t 4:0:0 --exclusive -A <YOUR_PROJECT>-SL3-GPU -p ampere` (exclusive access to a computing node)
or
`sintr -t 0:30:0 --gres=gpu:4  -A <YOUR_PROJECT>-SL3-GPU -p ampere` (inclusive access to a node with X GPUs).

CSD3 uses SLURM for job scheduling. To submit jobs to HPC, you will be required to modify the submission script, an example can be found at `/usr/local/Cluster-Docs/SLURM/slurm_submit.wilkes3` (for ampere) and `/usr/local/Cluster-Docs/SLURM/slurm_submit.wilkes2` (for pascal). 

Modify the options in both files as appropriate. Mostly, there are three sections:
- modify sbatch directives (#! sbatch directives begin here).
- load additional module (#! Insert additional module load commands after this line if needed).
- full path to application executable.

The differences between `slurm_submit.wilkes2` and `slurm_submit.wilkes3` are in the system. The rest are identical.
```
#SBATCH -p ampere
module load rhel8/default-amp 

#SBATCH -p pascal
module load rhel7/default-gpu
```

Some useful commands:
```
squeue      -> show global cluster information
sinfo       -> show global cluster information
sview       -> show global cluster information
scontrol show job nnnn -> examine the job with jobid nnnn
scontrol show node nodename -> examine the node with name nodename
sbatch      -> submits an executable script to the queueing system
sintr       -> submits an interactive job to the queueing system
srun        -> run a command either as a new job or within an existing job
scancel     -> delete a job
mybalance   -> show current balance of core hour credits
```

## 5. a simple example: submit `cifar10_tutorial` to wilkes2 (pascal) or wilkes3 (ampere)

Prerequisite: follow the same step of [Setup PyTorch on CSD3](https://docs.hpc.cam.ac.uk/hpc/software-packages/pytorch.html).

- terminal:
(1). modify sbatch directives and environment in `my_slurm_submit.wilkes2` or `my_slurm_submit.wilkes3` 
(2). execute `sbatch my_slurm_submit.wilkes2` or `sbatch my_slurm_submit.wilkes3`
(3). check results (as per your configuration, you will be notified via email when the execution commences, concludes, or encounters an error). The logs will be in the work directory and names as slurm-16363149 ($JOBID$).out.

- login-web.hpc.cam.ac.uk
(1) select Job -> Job Composer; 
(2) click `New Job` -> From Templates; select "Simple GPU Job"; change the Job Name;
(3) modify the Submit Script (slurm_submit.wilkes2); you can open the directory in the file editor (click Open Editor at the right bottom)
(4) click `submit` and the results will be presented on the right section.

You can use the [dashboard](https://login-web.hpc.cam.ac.uk/pun/sys/dashboard/activejobs) to monitor you submitted job.


useful docs (official instructions):
- Research Computing Services HPC Documentation: [link](https://docs.hpc.cam.ac.uk).
- Quickstart: [link](https://docs.hpc.cam.ac.uk/hpc/user-guide/quickstart.html).
- Frequently asked questions (FAQ): [link](https://docs.hpc.cam.ac.uk/hpc/user-guide/mfa.html).

some configuration
- UIS Password Management Application: [link](https://password.csx.cam.ac.uk).
- TOTP Verification Code. CSD3 and storage services logins require MultiFactor Authentication (MFA): [link](https://docs.hpc.cam.ac.uk/hpc/user-guide/mfa.html#walkthrough-ssh-to-multi-hpc-cam-ac-uk).
- CSD3 Host Keys: [link](https://docs.hpc.cam.ac.uk/hpc/user-guide/hostkeys.html).
- Research Storage Services (Login-Web Interface): [link](https://docs.hpc.cam.ac.uk/hpc/user-guide/login-web.html).

account
- full description of service levels (SLs): [link](https://docs.hpc.cam.ac.uk/hpc/user-guide/policies.html#service-levels).
- resource limits:
> On CPU, SL1 and SL2 users are limited to 4256 cores in use at any one time and a maximum wallclock runtime of 36 hours per job. 
> On GPU, SL1 and SL2 are limited to 64 GPUs in use at any one time and a maximum wallclock runtime of 36 hours per job. 
> SL3 users are similarly limited to 448 cores (CPU) and 32 GPUs (GPU), all with up to 12 hours per job. For more information, please see this full description of [service levels (SLs)](https://docs.hpc.cam.ac.uk/hpc/user-guide/policies.html#service-levels).

- The Wilkes3-GPU (ampere) nodes each contain 4 NVIDIA A100 GPUs. It is possible to request 1, 2, 3, or 4 GPUs for a single node job, but multinode jobs will need to request 4 GPUs per node, which would be done by using the directives

important notes
- purchase core hour credits (check `mybalance`);
- use ~/rds/ instead of ~/home/ as the personal "work" directory (RDS means Cambridge Research Data Storage);
- assess RHEL8 CPU cluster nodes and use `ampere (aka Wilkes3) GPU` compute nodes (4 A100). `RHEL8 logins are balanced over login-q-1 to login-q-4`;
- Note that the login-icelake nodes do not have GPUs and have a different (Intel) CPU type to the (AMD) ampere GPU nodes, so ampere development may require interactive use of a GPU node (see [here](https://docs.hpc.cam.ac.uk/hpc/user-guide/interactive.html#sintr) for more information].
