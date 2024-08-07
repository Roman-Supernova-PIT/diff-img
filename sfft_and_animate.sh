#!/bin/bash

#SBATCH --job-name=sfft
#SBATCH --mem=264G
##SBATCH --partition=cosmology
##SBATCH -w dcc-cosmology-07
#SBATCH --output=/hpc/group/cosmology/lna18/rsim_photometry/dcc_output/sfft/sfft-%J.out
#SBATCH --mail-user=lauren.aldoroty@duke.edu
#SBATCH --mail-type=ALL
#SBATCH --ntasks 5 # IMPORTANT: This is now the number of templates. 
#SBATCH --cpus-per-task 1
#SBATCH --array=1-7

# Need these lines if using GPU: 
##SBATCH -p gpu-common
##SBATCH --gres=gpu:1
##SBATCH --exclusive

# Activate conda environment
source ~/.bashrc
conda activate repeatability

# Run program. 
sne=( '20172782' )
# sne=( 20172117 20172782 20173305 20174023 20174118 20174370 20175077 20177380 20172328 20173301 20173373 20174108 20174213 20174542 20175568 20202893 )

for sn in "${sne[@]}"
do
    echo "$sn"
    srun python -u sfft_and_animate.py "$sn" --verbose True --slurm_array
done