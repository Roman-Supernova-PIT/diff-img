#!/bin/bash

#SBATCH -A m4385
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH --job-name=sfft
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/global/cfs/cdirs/m4385/users/lauren/out_logs/sfft/sfft-%J.out
#SBATCH --mail-user=lauren.aldoroty@duke.edu
#SBATCH --mail-type=ALL
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 1
#SBATCH --array=3

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
export SN_INFO_DIR="/pscratch/sd/l/laldorot/object_tables" # Location of object/image tables.
export SIMS_DIR="/global/cfs/cdirs/lsst/production/roman-desc-sims/Roman_data" # Location of the Roman-DESC sims.
export DIA_OUT_DIR="/pscratch/sd/l/laldorot/dia_out" # Parent output folder for DIA pipeline.

# Run program. 
sne=( '20172782' )
# sne=( '20172117' '20172782' '20173305' '20174023' '20174118' '20174370' '20175077' '20177380' '20172328' '20173301' '20173373' '20174108' '20174213' '20174542' '20175568' '20202893' )

for sn in "${sne[@]}"
do
    echo "$sn"
    # Step 1: Get all images the object is in.
    python -u get_object_instances.py "$sn"
    # Step 2: Sky subtract, align images to be in DIA. 
    srun python -u preprocess.py "$sn" --n-templates 1 --verbose True --slurm_array
    # WAIT FOR COMPLETION. 
    # Step 3: Get, align, save PSFs; cross-convolve. 
    # WAIT FOR COMPLETION.
    # Step 4: Differencing (GPU). 
    # WAIT FOR COMPLETION.
    # Step 5: Generate decorrelation kernel, apply to diff. image and science image, make stamps. 
    srun python -u sfft_and_animate.py "$sn" --n-templates 1 --verbose True --slurm_array
done