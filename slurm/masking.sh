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

module load anaconda
source /gpfs/projects/upf97/envs/mtg-bsc-wandb/bin/activate

python3 src/train.py --config cfg/config_masking.gin
