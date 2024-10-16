#!/bin/bash

#SBATCH --job-name=lc
#SBATCH --mem=8G
#SBATCH --output /hpc/group/cosmology/lna18/rsim_photometry/dcc_output/lc/lc-%j.out
##SBATCH --partition=cosmology
#SBATCH --mail-user=lauren.aldoroty@duke.edu
#SBATCH --mail-type=ALL
#SBATCH --array=2,4-6

# Activate conda environment
source ~/.bashrc
conda activate diff

python -u mk_lc.py 20172782 5 --inputdir /work/lna18/ --outputdir /hpc/group/cosmology/lna18/roman_sim_imgs/ --slurm_array