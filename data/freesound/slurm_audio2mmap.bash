#!/bin/bash
#SBATCH --partition=acc
#SBATCH --account=upf97
#SBATCH --qos=acc_resa
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --output=debug_%j_output.txt
#SBATCH --mail-type=all
#SBATCH --mail-user=dmitry.bogdanov@upf.edu

#RUN the du command
source /gpfs/projects/upf97/envs/essentia2/bin/activate

bash $1
