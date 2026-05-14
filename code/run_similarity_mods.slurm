#!/bin/bash
#SBATCH --job-name=similarity_mods
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/similarity_mods_%j.out
#SBATCH --error=slurm_logs/similarity_mods_%j.err

module load R/4.4.2

Rscript code/run_similarity_mods.R
