#!/bin/bash

#SBATCH --job-name s6
#SBATCH --account=upf97
#SBATCH --partition=acc
#SBATCH --qos=acc_resa
#SBATCH --nodes=1 # This needs to match Trainer(num_nodes=...)
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --output=debug_s6_%j_output.txt
#SBATCH --mail-type=all
#SBATCH --mail-user=pedro.ramoneda@upf.edu

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

source /gpfs/projects/upf97/envs/mtg-bsc/bin/activate

srun python3 src/train.py /gpfs/projects/upf97/logs/mtg-ssl/92n2bowt/checkpoints/config_masking_conformer_multiview_enc_to_auenccqt_small.gin