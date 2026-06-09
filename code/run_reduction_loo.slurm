#!/bin/bash
#SBATCH --job-name=reduction_loo
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=slurm_logs/reduction_loo_%j.out
#SBATCH --error=slurm_logs/reduction_loo_%j.err

module load R/4.4.2
export TBB_CXX_TYPE=gcc

Rscript code/run_reduction_loo.R
