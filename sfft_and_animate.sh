#!/bin/bash

#SBATCH --job-name=sfft
#SBATCH --mem=264G
#SBATCH --partition=cosmology
#SBATCH -w dcc-cosmology-03
#SBATCH --output=/hpc/group/cosmology/lna18/rsim_photometry/dcc_output/sfft/sfft-%J.out
#SBATCH --mail-user=lauren.aldoroty@duke.edu
#SBATCH --mail-type=ALL
#SBATCH --array=6

# Need these lines if using GPU: 
##SBATCH -p gpu-common
##SBATCH --gres=gpu:1
##SBATCH --exclusive

# Activate conda environment
source ~/.bashrc
conda activate repeatability

# Run program. 
# As job array:
python -u sfft_and_animate.py 20172782 1 --slurm_array

# Loop through bands:
# bands=( R062 Z087 Y106 J129 H158 F184 K213 )

# for j in "${bands[@]}"
# do
#     echo 20172117
#     echo "$j"
#     python -u sfft_and_animate.py 20172117 "$j"
# done