#!/bin/bash
#SBATCH --partition=acc
#SBATCH --account=upf97
#SBATCH --qos=acc_resa
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00:00
#SBATCH --output=debug_$1_output.txt
#SBATCH --mail-type=all
#SBATCH --mail-user=pedro.ramoneda@upf.edu

#RUN the du command
module load anaconda
source ../../essentia/bin/activate

python3 subset_audio2mmap.py $1
