#!/bin/bash

#SBATCH --job-name="mnist"
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --account=Education-EEMCS-Courses-CSE3000
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=2
#SBATCH --mem-per-cpu=4G

srun python main.py > mnist.log