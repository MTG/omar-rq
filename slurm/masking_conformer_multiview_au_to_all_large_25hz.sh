#!/bin/bash

#SBATCH --job-name a2a_large_25hz
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
# interrrupt and resubmit 90 seconds before training ends (experimental)
# https://pytorch-lightning.readthedocs.io/en/1.2.10/clouds/slurm.html#wall-time-auto-resubmit
#SBATCH --signal=SIGUSR1@90

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

source /gpfs/projects/upf97/envs/mtg-bsc/bin/activate

srun python3 src/train.py cfg/config_masking_conformer_multiview_au_to_all_large_25hz.gin
