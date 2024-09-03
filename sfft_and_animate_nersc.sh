#!/bin/bash

#SBATCH -A m4385
#SBATCH -C gpu
#SBATCH -q debug # regular or debug
#SBATCH --job-name=sfft
##SBATCH --mem=64G # Comment this out while on -q debug because you get a whole node, so you can monitor with top to see how much you're using. 
#SBATCH --ntasks=1 # Leave as 1 because the tasks are divided in python. 
#SBATCH --ntasks-per-node=1 # Same here. Leave as 1. 
#SBATCH --output=/global/cfs/cdirs/m4385/users/lauren/out_logs/sfft/sfft-%J.out
#SBATCH --mail-user=lauren.aldoroty@duke.edu
#SBATCH --mail-type=ALL
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 1
##SBATCH --time=2:00:00 # regular QOS
#SBATCH --time=30:00 # debug QOS
#SBATCH --array=5

# Activate conda environment
source ~/.bashrc
conda activate diff
module load cudatoolkit/12.2 # Default on NERSC, but just in case... be explicit. 

# Make numpy stop thread hogging
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Set environment variables
export SN_INFO_DIR="/pscratch/sd/l/laldorot/object_tables" # Location of object/image tables.
export SIMS_DIR="/global/cfs/cdirs/lsst/shared/external/roman-desc-sims/Roman_data" # Location of the Roman-DESC sims.
export SNANA_PQ_DIR="/global/cfs/cdirs/lsst/www/DESC_TD_PUBLIC/Roman+DESC/PQ+HDF5_ROMAN+LSST_LARGE" # Location of the SNANA parquet files. 
export SNID_LC_DIR="/global/cfs/cdirs/lsst/www/DESC_TD_PUBLIC/Roman+DESC/ROMAN+LSST_LARGE_SNIa-normal" # Path to LC truth files
export DIA_OUT_DIR="/pscratch/sd/l/laldorot/dia_out" # Parent output folder for DIA pipeline.
export NVCC="/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/bin/nvcc" # `whereis nvcc` path
export LC_OUT_DIR="/global/cfs/cdirs/m4385/users/lauren/lcs" # Parent folder of save location for photometry data and figures.

# Run program. 
sne=( '20172782' )
# sne=( '20172117' '20172782' '20173305' '20174023' '20174118' '20174370' '20175077' '20177380' '20172328' '20173301' '20173373' '20174108' '20174213' '20174542' '20175568' '20202893' )
# sne=( '20175077' '20177380' '20172328' '20173301' '20173373' '20174108' '20174213' '20174542' '20175568' '20202893' )

for sn in "${sne[@]}"
do
    echo "$sn"
    # Step 1: Get all images the object is in.
    # python -u get_object_instances.py "$sn" # We actually only want to do this step once per object.
                                            # There is no reason to do it once per job array
                                            # element (i.e., once per object per band).
    # Step 2: Sky subtract, align images to be in DIA. 
    # WAIT FOR COMPLETION. 
    # Step 3: Get, align, save PSFs; cross-convolve. 
    srun python -u preprocess.py "$sn" --n-templates 1 --verbose True --slurm_array
    # WAIT FOR COMPLETION.
    # Step 4: Differencing (GPU). 
    srun python -u sfftdiff.py "$sn" --n-templates 1 --verbose True --slurm_array
    # WAIT FOR COMPLETION.
    # Step 5: Generate decorrelation kernel, apply to diff. image and science image, make stamps. 
    srun python -u postprocess.py "$sn" --n-templates 1 --verbose True --slurm_array
    # WAIT FOR COMPLETION.
    # Step 6: Make LC and generate plots.
    python -u mk_lc.py "$sn" --n-templates 1 --slurm_array
done