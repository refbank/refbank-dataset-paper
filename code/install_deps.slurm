#!/bin/bash
#SBATCH --job-name=install_deps
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/install_deps_%j.out
#SBATCH --error=slurm_logs/install_deps_%j.err

module load R/4.4.2
export TBB_CXX_TYPE=gcc

Rscript code/install_deps.R
