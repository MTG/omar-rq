#!/bin/bash
#SBATCH -J try
#SBATCH -N 1
#SBATCH --account=upf97
#SBATCH --partition=acc
#SBATCH --qos=acc_resa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=4:00:00
#SBATCH --output=debug_structure_%j_output.txt
#SBATCH --mail-type=all
#SBATCH --mail-user=pedro.ramoneda@upf.edu

module load anaconda
source /gpfs/projects/upf97/envs/mtg-bsc/bin/activate

ssl_model_id=$1
downstream_config=$2

# example: python src/downstream.py o8tkup9f cfg/downstream/structure.gin
# example: python src/downstream.py ws3pyty7 cfg/downstream/structure.gin
# example: python src/downstream.py pyypqq7g cfg/downstream/structure.gin
# sbatch slurm/downstream.sh pyypqq7g cfg/downstream/structure.gin
python src/downstream.py $1 $2