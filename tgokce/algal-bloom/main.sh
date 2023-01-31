#!/bin/bash

#SBATCH --job-name="algal-bloom"
#SBATCH --account=Education-EEMCS-Courses-CSE3000
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem=32G

srun python main.py > output.log