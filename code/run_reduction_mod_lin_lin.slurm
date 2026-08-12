#!/bin/bash
#SBATCH --job-name=red_mod_lin_lin
#SBATCH --partition=owners,mcfrank
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/red_mod_lin_lin_%j.out
#SBATCH --error=slurm_logs/red_mod_lin_lin_%j.err

module load R/4.4.2

export R_LIBS_USER="$HOME/.Rlibs/refbank-dataset-paper/4.4.2"
mkdir -p "$R_LIBS_USER"
export TBB_CXX_TYPE=gcc

Rscript code/run_reduction_mod_lin_lin.R
