#!/bin/bash
#SBATCH --job-name=red_pred_all_moderators
#SBATCH --partition=owners,mcfrank
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/red_pred_all_moderators_%j.out
#SBATCH --error=slurm_logs/red_pred_all_moderators_%j.err

module load R/4.4.2
export TBB_CXX_TYPE=gcc

Rscript code/run_reduction_pred_mod_all_moderators.R
