#!/bin/bash
#SBATCH -J compute_encodec_stats
#SBATCH -N 1
#SBATCH --account=upf97
#SBATCH --partition=acc
#SBATCH --qos=acc_resa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --time=10:00:00
#SBATCH --output=debug_%j_output.txt
#SBATCH --mail-type=all
#SBATCH --mail-user=pablo.alonso@upf.edu

source /gpfs/projects/upf97/envs/mtg-bsc/bin/activate


srun python3 src/compute_input_stats.py cfg/config_conformer_encodec.gin
