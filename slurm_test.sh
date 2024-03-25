#!/bin/sh
#SBATCH --job-name=covr
#SBATCH --account=<account_name>
#SBATCH --partition=<partition>
#SBATCH --time=3-00:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=4

srun python test.py test=webvid-covr model/ckpt=webvid-covr-test