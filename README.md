# Lauren's Roman forced photometry pipeline. 

Runs on Troxel et al. RomanDESCSims simulated Roman WFI images.
Currently developed and tested at Duke Computing Cluster and Perlmutter at NERSC.

## Depends on   
`phrosty`  
`roman_imsim`  

## Runs with 
1. `get_object_instances.py` to get all the images by filter, pointing, and SCA that contain the specified transient.
2. `preprocess.py` to create references, get PSFs, and do cross-convolutions.
3. `sfftdiff.py` to run SFFT subtraction on a GPU.
4. `postprocess.py` to generate decorrelation kernels, apply those to images, and make stamps.
5. `mk_lc.py` to extract a lightcurve from the subtractions.


### sfft_and_animate_*

`sfft_and_animate_*.sh` as a SLURM array job that has SN ID hardcoded.  
This then calls `get_object_instances`, `preprocess`, `sfftdiff`, `postprocess`, and `mk_lc` with the SN ID (refered to as `oid` in the code).  
`sfft_and_animate_*.py` reads the SLURM_ARRAY_JOBID from the environment and uses that to select the band to process.  (1-7).

`sfft_and_animate_*.py` reads input data from input data as defined by `phrosty.utils._build_filepath`, which uses environment variable`SIMS_DIR` to look for the data. `SIMS_DIR` should match where the RomanDESCSims are.

`preprocess.py`, `sfftdiff.py`, and `postprocess.py` write files out to environment variable `DIA_OUT_DIR`.

`mk_lc.py` writes files out to environment variable `LC_OUT_DIR`.

Example usage at the DCC
Run on SNID 20172782 (yes, the SN IDs are 8-digit numbers beginning with 20, they are not dates), filter R062, with specified input directory.
```
python mk_lc.py 20172782 --band R062 --inputdir /work/lna18/
```

Example usage on the Roman Science Platform to set things up to subtract.
```
oid=20000808
band=H158
python get_object_instances.py ${oid} 
python preprocess.py ${oid} --band ${band}
```
