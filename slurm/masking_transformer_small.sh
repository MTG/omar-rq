#!/bin/bash

#SBATCH --job-name masking_conformer_small
#SBATCH --account=upf97
#SBATCH --partition=acc
#SBATCH --qos=acc_resa
#SBATCH --nodes=1 # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --time=72:00:00
#SBATCH --output=debug_%j_output.txt
#SBATCH --mail-type=all
#SBATCH --mail-user=pedro.ramoneda@upf.edu

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

source /gpfs/projects/upf97/envs/mtg-bsc/bin/activate

srun python3 src/train.py cfg/config_masking_transformer_small.gin

