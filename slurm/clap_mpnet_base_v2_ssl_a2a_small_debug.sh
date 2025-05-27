#!/bin/bash

#SBATCH --job-name tam_debug
#SBATCH --account=upf97
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=debug_%j_output.txt

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

source /gpfs/projects/upf97/envs/mtg-bsc/bin/activate
srun python src/train_clap.py cfg/config_clap_mpnet_base_v2_ssl_a2a_small_debug.gin
