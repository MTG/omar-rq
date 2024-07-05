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

source /etc/profile.d/lmod.sh
module load Python/3.8.6-GCCcore-10.2.0
module load CUDA/12.1.0
pip3 install --upgrade pip

pip3 install torch torchvision torchaudio
pip3 install git+https://github.com/huggingface/transformers.git@main bitsandbytes
pip3 install git+https://github.com/huggingface/accelerate.git
pip3 install git+https://github.com/huggingface/peft.git@e536616888d51b453ed354a6f1e243fecb02ea08
pip3 install scikit-learn tensorboard pandas matplotlib music21 seaborn tqdm
pip3 install wandb==0.13.3

export WANDB_API_KEY=0bd2bcedf5cb93b5aa685f6fc7192aba6fd7b74b
python -m wandb login

python3 train_codellama_remi.py