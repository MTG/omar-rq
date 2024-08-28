#!/bin/bash
#SBATCH --job-name masking_conformer_big
#SBATCH --account=upf97
#SBATCH --partition=acc
#SBATCH --qos=acc_resa
#SBATCH --nodes=2 # This needs to match Trainer(num_nodes=...)
#SBATCH --ntasks-per-node=4 # This needs to match Trainer(devices=...)
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --output=debug_%j_output.txt
#SBATCH --mail-type=all
#SBATCH --mail-user=pablo.alonso@upf.edu

module load anaconda
source /gpfs/projects/upf97/envs/mtg-bsc/bin/activate


srun python3 src/train.py cfg/config_masking_conformer_big.gin
