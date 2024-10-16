#!/bin/bash
#SBATCH --job-name=7b_olmo_1n4g
#SBATCH --account=<account_name>
#SBATCH --output /path/to/your/run_results/%x_%j/output_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH --error /path/to/your/run_results/%x_%j/error_%j.out  # File to which STDERR will be written, %j inserts jobid
#SBATCH --nodes=1              # Total number of nodes
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4       # Allocate one gpu per MPI rank
#SBATCH --cpus-per-task=24
#SBATCH --time=2:00:00
#SBATCH --mem=0			# All memory on the node
#SBATCH --exclusive
#SBATCH --partition=kempner_h100

module load python/3.10.13-fasrc01 cuda/12.2.0-fasrc01 cudnn/8.9.2.26_cuda12-fasrc01
conda deactivate
conda activate /path/to/your/PLMo/conda-environment
nvidia-smi
echo $CONDA_PREFIX

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPICH_GPU_SUPPORT_ENABLED=1

export PYTHONPATH=.:${PYTHONPATH}

#Path to the folder to save the checkpoints
export CHECKPOINTS_PATH=/path/to/your/run_results/${SLURM_JOB_NAME}_${SLURM_JOB_ID}/checkpoints

export PYTORCH_KERNEL_CACHE_PATH=/tmp/pytorch_kernel_cache/
mkdir -p $PYTORCH_KERNEL_CACHE_PATH


srun \
  --cpus-per-task=$SLURM_CPUS_PER_TASK \
  --distribution=block:block \
  --kill-on-bad-exit \
  scripts/run_with_environment.sh \
    python -u scripts/train.py configs/kempner_institute/7b_Olmo.yaml \
      --run_name=${SLURM_JOB_NAME}_${SLURM_JOB_ID} \
      --data.num_workers=16 \
      --data.prefetch_factor=4 \
      --model.flash_attention=false \
      ${@}
