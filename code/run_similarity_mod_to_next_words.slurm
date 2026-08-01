#!/bin/bash
#SBATCH --job-name=sim_to_next_words
#SBATCH --partition=owners,mcfrank
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/sim_to_next_words_%j.out
#SBATCH --error=slurm_logs/sim_to_next_words_%j.err

module load R/4.4.2
export TBB_CXX_TYPE=gcc

Rscript code/run_similarity_mod_to_next_words.R
