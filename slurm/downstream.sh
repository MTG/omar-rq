#!/bin/bash
#SBATCH -J try
#SBATCH -N 1
#SBATCH --account=upf97
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:60:00
#SBATCH --output=debug_%j_output.txt

module load anaconda
source /gpfs/projects/upf97/envs/mtg-bsc/bin/activate

ssl_model_id=$1
downstream_config=$2

# example: python src/downstream.py ws3pyty7 cfg/downstream/structure_local.gin
python src/downstream.py $1 $2