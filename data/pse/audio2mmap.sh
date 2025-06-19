#!/bin/bash

#SBATCH --partition=acc
#SBATCH --account=upf97
#SBATCH --qos=acc_resa
#SBATCH --array=0-79
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00:00
#SBATCH --output=pse_dataset_debug_output.txt
#SBATCH --mail-type=all
#SBATCH --mail-user=pablo.alonso@upf.edu

source /gpfs/projects/upf97/envs/essentia2/bin/activate

srun python3 audio2mmap.py --n-tasks 80 --task-id $SLURM_ARRAY_TASK_ID --input-dir "/gpfs/projects/upf97/pse" --output-dir "/gpfs/scratch/upf97/mmaps_pse/"
