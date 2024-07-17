#!/bin/bash

#SBATCH --job-name=lc
#SBATCH --mem=8G
#SBATCH --output /hpc/group/cosmology/lna18/rsim_photometry/dcc_output/lc/lc-%j.out
#SBATCH --partition=cosmology
#SBATCH --mail-user=lauren.aldoroty@duke.edu
#SBATCH --mail-type=ALL
#SBATCH --array=5

# Activate conda environment
source ~/.bashrc
conda activate repeatability

python -u mk_lc.py 20172782