#!/bin/bash

#SBATCH --job-name clap_mpnet_base_v2_ssl_a2a_small
#SBATCH --account=upf97
#SBATCH --partition=acc
#SBATCH --qos=acc_resa
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --output=debug_%j_output.txt
#SBATCH --mail-type=all
#SBATCH --mail-user=pablo.alonso@upf.edu

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

source /gpfs/projects/upf97/envs/mtg-bsc/bin/activate
srun python src/train_clap.py cfg/text_audio/config_clap_mpnet_base_v2_ssl_a2a_small.gin
