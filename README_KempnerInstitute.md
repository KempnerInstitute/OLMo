# OLMo On The Kempner Institute HPC Cluster

This section has been added to OLMo `README` page for Kempner Community to help them with a step by step guideline to run OLMo on the HPC cluster and start exploring.

## 1. Installation on The HPC Cluster 

### 1.1. Create Conda Environment and Activate it

First create your conda environment using one of the following commands:

Load the python module first and then create the conda environment.

```bash
module load python/3.10.13-fasrc01
```

Create a conda environment to hold your Olmo installation.
You can name it `olmo_test` and add it in the conda environment default path:

```bash
mamba create --name olmo_test python=3.10
```

Or if you wish to add it into a specific path you can use the following: 

```bash
mamba create --prefix [full_path_to_your_env] python=3.10
```

We recommend using `mamba` instead of `conda` for faster installation. We recommend using a 
prefix path as well as it makes sharing the environment with others easier.

Activate the environment using:

```bash
conda activate olmo_test
```

Or if you use another path

```bash
conda activate [path_to_your_env]
```

### 1.2. Install PyTorch

First install [PyTorch](https://pytorch.org) according to the instructions specific to your operating system. For installation 
on the linux cluster using `mamba` and CUDA 12.4 (Cuda >=11.9 is required for H100 GPUS) you can use the following command:

```bash
mamba install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

### 1.3. Install OLMo from Source Code 

To install from source (recommended for training/fine-tuning) run:

```bash
git clone https://github.com/KempnerInstitute/OLMo.git
cd OLMo
pip3 install -e .[all]
```

## 2. Run OLMo on the Kempner Institute HPC Cluster

Now that we have the Conda environment ready, it's time to run OLMo. To do that, we need a config file to pass it to the training script to specify all OLMo configs and hyperparameters as well as a SLURM script to submit the job on the HPC cluster. 

### 2.1. Config file

Two config files have been provided by which an OLMo model is trained on 4 GPUs using the `c4` data which is tokenized by `t5-base` tokenizer. You can take these config files and may adjust its different hyperparameters based on your need. These config files are as follows:

* [configs/kempner_institute/7b_Olmo.yaml](configs/kempner_institute/7b_Olmo.yaml) which enables running of a 7b-parameter OLMo model on 4 H100 gpus on a single node using `FSDP`
* [configs/kempner_institute/1b_Olmo.yaml](configs/kempner_institute/1b_Olmo.yaml) which enables running of a 1b-parameter OLMo model on 4 H100 gpus on a single node using `DDP`

Note that you should at least modify the `wandb` section of the config file according to your `wandb` account and also setup your `wandb` account on the cluster if you haven't already. You may also simply comment out the `wandb` section on the config file if you dont wish to use `wandb` for logging.

```{code} bash
wandb:
  name: ${run_name}
  entity: <entity_name>
  project: <project_name>
```

### 2.2. SLURM Script

To run OLMo on the HPC cluster using SLURM, you may use the SLURM script skeleton in [scripts/kempner_institute/submit_srun.sh](scripts/kempner_institute/submit_srun.sh). This will run OLMo using 4 H100 GPUs on a single node.
Note that the following items should be updated in the above SLURM script skeleton:

* `#SBATCH --job-name=<job-name>`       - Name for your submitting job - default: `run_olmo_1n4g` (`1n4g` stands for 4 gpus on the same node)
* `#SBATCH --account=<account_name>`    - Account name to use the cluster
* `#SBATCH --output <output_path>`      - File to which STDOUT will be written - default: `./<job-name>_<job-id>/output_<job-id>.out` in the current directory (`job-id` will be assigned by SLURM)
* `#SBATCH --error <error_output_path>` - File to which STDERR will be written - default: `./<job-name>_<job-id>/error_<job-id>.out` in the current directory
* `conda activate </path/to/your/OLMo/conda-environment>` - Activate conda environment that you just created - by default it activates conda environment named `olmo_test` from the default conda path.  
* `export CHECKPOINTS_PATH=</path/to/save/checkpoints`    - Path to the folder to save the checkpoints - default: `./<job-name>_<job-id>/checkpoints`
* `python -u scripts/train.py <config_file>` - Pass in either 7b_Olmo.yaml or 1b_Olmo.yaml config files to the train.py (by default it will run 7b OLMo using FSDP you can change the input config file to `configs/kempner_institute/1b_Olmo.yaml` in order to run 1b OLMo using DDP).

Submit the SLURM job by the following command.

```bash
sbatch scripts/kempner_institute/submit_srun.sh
```