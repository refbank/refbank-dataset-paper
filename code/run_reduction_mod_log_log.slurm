#!/bin/bash
#SBATCH --job-name=red_mod_log_log
#SBATCH --partition=owners,mcfrank
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/red_mod_log_log_%j.out
#SBATCH --error=slurm_logs/red_mod_log_log_%j.err

module load R/4.4.2

export TBB_CXX_TYPE=gcc

Rscript code/run_reduction_mod_log_log.R
