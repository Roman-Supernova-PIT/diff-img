import os

sne = [20001783, 20007992, 20012160, 20015318, 20022131, 20027046, 20029396, 20035246, 20037013, 20044575, 20045257, 20046134, 20046407, 20047507, 20049862, 20064501, 20066972, 20067672, 20067866, 20086185]
n_templates = 1

def mk_bash(sn,n_templates,jobname='sfft'):
    bash_head = '#!/bin/bash \n' + \
                '\n' +  \
                '#SBATCH -A m4385 \n' +  \
                '#SBATCH -C gpu \n' + \
                '#SBATCH -q regular # regular or debug  \n' +  \
                f'#SBATCH --job-name={jobname}_{n_templates}  \n' +  \
                '##SBATCH --mem=64G # Comment this out while on -q debug because you get a whole node, so you can monitor with top to see how much youre using.  \n' +  \
                '#SBATCH --ntasks=1 # Leave as 1 because the tasks are divided in python.  \n' +  \
                '#SBATCH --ntasks-per-node=1 # Same here. Leave as 1.  \n' +  \
                '#SBATCH --output=/global/cfs/cdirs/m4385/users/lauren/out_logs/sfft/sfft-%J.out \n' +  \
                '#SBATCH --mail-user=lauren.aldoroty@duke.edu \n' +  \
                '#SBATCH --mail-type=ALL \n' +  \
                '#SBATCH --gpus-per-task 1 \n' +  \
                '#SBATCH --cpus-per-task 1 \n' +  \
                '#SBATCH --time=2:00:00 # regular QOS \n' +  \
                '#SBATCH --array=1-7 \n' +  \
                '\n' +  \
                '# Activate conda environment \n' +  \
                'source ~/.bashrc \n' +  \
                'conda activate diff \n' +  \
                'module load cudatoolkit/12.2 # Default on NERSC, but just in case... be explicit.  \n' +  \
                '\n' +  \
                '# Make numpy stop thread hogging \n' +  \
                'export OPENBLAS_NUM_THREADS=1 \n' +  \
                'export MKL_NUM_THREADS=1 \n' +  \
                'export NUMEXPR_NUM_THREADS=1 \n' +  \
                'export OMP_NUM_THREADS=1 \n' +  \
                'export VECLIB_MAXIMUM_THREADS=1 \n' +  \
                '\n' +  \
                '# Set environment variables \n' +  \
                'export SN_INFO_DIR="/pscratch/sd/l/laldorot/object_tables" # Location of object/image tables. \n' +  \
                'export SIMS_DIR="/global/cfs/cdirs/lsst/shared/external/roman-desc-sims/Roman_data" # Location of the Roman-DESC sims. \n' +  \
                'export SNANA_PQ_DIR="/global/cfs/cdirs/lsst/www/DESC_TD_PUBLIC/Roman+DESC/PQ+HDF5_ROMAN+LSST_LARGE" # Location of the SNANA parquet files.  \n' +  \
                'export SNID_LC_DIR="/global/cfs/cdirs/lsst/www/DESC_TD_PUBLIC/Roman+DESC/ROMAN+LSST_LARGE_SNIa-normal" # Path to LC truth files \n' +  \
                'export DIA_OUT_DIR="/pscratch/sd/l/laldorot/dia_out" # Parent output folder for DIA pipeline. \n' +  \
                'export NVCC="/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/bin/nvcc" # `whereis nvcc` path \n' +  \
                'export LC_OUT_DIR="/global/cfs/cdirs/m4385/users/lauren/lcs" # Parent folder of save location for photometry data and figures. \n' \
                '\n' + \
                '# How many templates am I using? \n' +  \
                f'n_templates=\"{n_templates}\"\n' + \
                f'sn=\"{sn}\"'

    bash_main = '\n' + \
                'echo \"$sn\" \n' + \
                '# Step 1: Get all images the object is in. \n' + \
                '# We actually only want to do this step once per object. \n' + \
                '# There is no reason to do it once per job array \n' + \
                '# element (i.e., once per object per band). \n' + \
                f'python -u get_object_instances.py \"$sn\" --n-templates \"$n_templates\" \n' + \
                '\n' + \
                '# Step 2: Sky subtract, align images to be in DIA. \n' + \
                '# WAIT FOR COMPLETION. \n' + \
                '# Step 3: Get, align, save PSFs; cross-convolve. \n' + \
                'python -u preprocess.py \"$sn\" --verbose True \n' + \
                '       --sci-list-path /pscratch/sd/l/laldorot/object_tables/\"$sn\"/\"$sn\"_instances_science.csv \n' + \
                '       --template-list-path /pscratch/sd/l/laldorot/object_tables/\"$sn\"/\"$sn\"_instances_template_\"$n_templates\".csv \n' + \
                '       --slurm_array \n' + \
                '# WAIT FOR COMPLETION. \n' + \
                '\n' + \
                '# Step 4: Differencing (GPU). \n' + \
                'python -u sfftdiff.py \"$sn\" --verbose True \n' + \
                '       --sci-list-path /pscratch/sd/l/laldorot/object_tables/\"$sn"/\"$sn"_instances_science.csv \n' + \
                '       --template-list-path /pscratch/sd/l/laldorot/object_tables/\"$sn\"/\"$sn\"_instances_template_\"$n_templates\".csv \n' + \
                '       --slurm_array \n' + \
                '# WAIT FOR COMPLETION. \n' + \
                '\n' + \
                '# Step 5: Generate decorrelation kernel, apply to diff. image and science image, make stamps. \n' + \
                'python -u postprocess.py \"$sn\" --verbose True \n' + \
                '       --sci-list-path /pscratch/sd/l/laldorot/object_tables/\"$sn\"/\"$sn\"_instances_science.csv \n' + \
                '       --template-list-path /pscratch/sd/l/laldorot/object_tables/\"$sn\"/\"$sn\"_instances_template_\"$n_templates\".csv \n' + \
                '       --slurm_array \n' + \
                '# WAIT FOR COMPLETION. \n' + \
                '\n' + \
                '# Step 6: Make LC and generate plots. \n' + \
                'python -u mk_lc.py \"$sn\" --verbose True \n' + \
                '--sci-list-path /pscratch/sd/l/laldorot/object_tables/\"$sn"/\"$sn\"_instances_science.csv \n' + \
                '--template-list-path /pscratch/sd/l/laldorot/object_tables/\"$sn\"/\"$sn\"_instances_template_\"$n_templates\".csv \n' + \
                '--slurm_array'

    bash_string = bash_head + bash_main
    return bash_string

for sn in sne:
    with open(f'{sn}_job.sh', 'w') as f:
        f.write(mk_bash(sn,n_templates,sn))
        
    os.system(f'sbatch {sn}_job.sh') 