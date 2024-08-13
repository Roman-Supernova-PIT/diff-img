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

# Run program. 
sne=( '20172782' )
# sne=( '20172117' '20172782' '20173305' '20174023' '20174118' '20174370' '20175077' '20177380' '20172328' '20173301' '20173373' '20174108' '20174213' '20174542' '20175568' '20202893' )

for sn in "${sne[@]}"
do
    echo "$sn"
    srun python -u sfft_and_animate.py "$sn" --n-templates 1 --verbose True --slurm_array
done