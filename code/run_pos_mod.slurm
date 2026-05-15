#!/bin/bash
#SBATCH --job-name=pos_mod
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/pos_mod_%j.out
#SBATCH --error=slurm_logs/pos_mod_%j.err

module load R/4.4.2
export TBB_CXX_TYPE=gcc

Rscript code/run_pos_mod.R
