#!/bin/bash
#SBATCH --job-name=install_deps
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=install_deps_%j.out
#SBATCH --error=install_deps_%j.err

module load R/4.4.2

Rscript code/install_deps.R
