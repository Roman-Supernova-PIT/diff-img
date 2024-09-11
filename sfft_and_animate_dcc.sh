#!/bin/bash

#SBATCH --job-name=sfft
#SBATCH --mem=264G # For serial jobs
#SBATCH --partition=cosmology
#SBATCH -w dcc-cosmology-05
#SBATCH --output=/hpc/group/cosmology/lna18/rsim_photometry/dcc_output/sfft/sfft-%J.out
#SBATCH --mail-user=lauren.aldoroty@duke.edu
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task 1
#SBATCH --array=1

# Need these lines if using GPU: 
##SBATCH -p gpu-common
##SBATCH --gres=gpu:1
##SBATCH --exclusive

# Activate conda environment
source ~/.bashrc
conda activate diff

# Make numpy stop thread hogging
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Set environment variables
export SN_INFO_DIR="/hpc/group/cosmology/lna18/roman_sim_imgs/Roman_Rubin_Sims_2024" # Location of object/image tables.
export SIMS_DIR="/cwork/mat90/RomanDESC_sims_2024" # Location of the Roman-DESC sims.
export DIA_OUT_DIR="/work/lna18/imsub_out" # Parent output folder for DIA pipeline.

# Run program. 
sne=( '20172782' )
# sne=( '20172117' '20172782' '20173305' '20174023' '20174118' '20174370' '20175077' '20177380' '20172328' '20173301' '20173373' '20174108' '20174213' '20174542' '20175568' '20202893' )

for sn in "${sne[@]}"
do
    echo "$sn"
    python -u get_object_instances.py "$sn"
    # srun python -u sfft_and_animate.py "$sn" --n-templates 1 --verbose True --slurm_array
done