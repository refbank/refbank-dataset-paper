#!/bin/bash
#SBATCH --job-name=sim_diverge
#SBATCH --partition=owners,mcfrank
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/sim_diverge_%j.out
#SBATCH --error=slurm_logs/sim_diverge_%j.err

module load R/4.4.2

export TBB_CXX_TYPE=gcc

Rscript code/run_similarity_mod_diverge.R
