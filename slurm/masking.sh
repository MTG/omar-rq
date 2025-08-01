#!/bin/bash
#SBATCH -J masking
#SBATCH -N 1
#SBATCH --account=upf97
#SBATCH --partition=acc
#SBATCH --qos=acc_resa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --time=72:00:00
#SBATCH --output=debug_%j_output.txt
#SBATCH --mail-type=all
#SBATCH --mail-user=pedro.ramoneda@upf.edu

source /gpfs/projects/upf97/envs/mtg-bsc/bin/activate

srun --cpus-per-task=80 python3 src/train.py cfg/config_masking.gin

