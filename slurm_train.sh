#!/bin/sh
#SBATCH --job-name=covr
#SBATCH --account=<account_name>
#SBATCH --partition=<partition>
#SBATCH --time=3-00:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gpus=a100-sxm4-80gb:8

srun python train.py