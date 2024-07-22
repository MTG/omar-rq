#!/bin/bash
#SBATCH -J pedrollama
#SBATCH -p high
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64g
#SBATCH --time=150:00:00
#SBATCH -o %N.%J.OUTPUT.out
#SBATCH -e %N.%J.ERROR_LOGS.err

module load anaconda
source mtg-bsc/bin/activate

python3 src/train.py --config cfg/config.gin
