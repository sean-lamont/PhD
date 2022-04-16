#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=nodup
#SBATCH --mem=8gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=wu099@csiro.au
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --output=log/train/nodup_%j.log   # Standard output and error log

module load python/3.7.2
module load pytorch/1.4.0-py37-cuda90
cd ~/temp/
source venv/bin/activate
cd RLHOL4/holgym
# echo "batch $1"
python nodup.py
