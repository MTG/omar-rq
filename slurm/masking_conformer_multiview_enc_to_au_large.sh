#!/bin/bash

#SBATCH --job-name e2w
#SBATCH --account=upf97
#SBATCH --partition=acc
#SBATCH --qos=acc_resa
#SBATCH --nodes=2 # This needs to match Trainer(num_nodes=...)
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --time=72:00:00
#SBATCH --output=debug_%j_output.txt
#SBATCH --mail-type=all
#SBATCH --mail-user=pablo.alonso@upf.edu

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

source /gpfs/projects/upf97/envs/mtg-bsc/bin/activate


srun python3 src/train.py cfg/config_masking_conformer_multiview_enc_to_au_large.gin
