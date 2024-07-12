#!/bin/bash
#SBATCH --partition=interactive
#SBATCH --account=upf97
#SBATCH --qos=acc_interactive
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-socket=20
#SBATCH --time=00:15:00
#SBATCH --output=du_output.txt

# Load any necessary modules
# module load <module_name>

# Run the du command
module load anaconda
source mtg-bsc/bin/activate

python mmap_dir.py $1 $2