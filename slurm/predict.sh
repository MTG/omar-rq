#!/bin/bash
#SBATCH -J try
#SBATCH -N 1
#SBATCH --account=upf97
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:30:00
#SBATCH --output=debug_%j_output.txt

module load anaconda
source /gpfs/projects/upf97/envs/mtg-bsc/bin/activate

train_cfg=$1
predict_cfg=$2

# e.g. python3 src/predict.py /gpfs/projects/upf97/logs/mtg-ssl/ws3pyty7/checkpoints/config_conformer.gin cfg/downstream/structure.gin
# e.g. python3 src/predict.py /gpfs/projects/upf97/logs/mtg-ssl/ws3pyty7/checkpoints/config_conformer.gin cfg/downstream/structure.gin


python3 src/predict.py $train_cfg $predict_cfg
