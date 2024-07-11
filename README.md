# Lauren's Roman forced photometry pipeline. 

Runs on Troxel et al. RomanDESCSims simulated Roman WFI images.
Currently developed and tested at Duke Computing Cluster.

## Depends on   
`phrosty`  
`roman_imsim`  

## Runs with 
1. `sfft_and_animate` to create references, run the subtractions, and create gifs.
2. `mk_lc.py` to extra a lightcurve from the subtractions.


### sfft_and_animate

`sfft_and_animate.sh` as a SLURM array job that has SN ID hardcoded.  
This then calls `sfft_and_animate.py` with the SN ID (refered to as `oid` in the code).  
`sfft_and_animate.py` reads the SLURM_ARRAY_JOBID from the environment and uses that to select the band to process.  (1-7).

`sfft_and_animate.py` reads input data from input data as defined by `phrosty.utils._build_filepath`, which uses environment `ROOTDIR` or a default to look for the data.  `ROOTDIR` should match where the RomanDESCSims are.

`sfft_and_animate.py` writes files out to `/work/lna18/imsub_out`.  This could be generalized to an `OUTDIR` or some more generic variable.

### mk_lc.py

`mk_lc` is structured similarly to `sfft_and_animate`

`mk_lc.sh` is a SLURM array job that has SN ID hardcoded.
This then calls `mk_lc.py` with the SN ID (refered to as `oid` in the code).
`mk_lc.py` read the SLURM_ARRAY_JOBID from the environment and uses that to select the band to process (1-7).

`mk_lc.py` takes the output from `sfft_and_animate.py` as input and makes a lightcurve from the subtracted images.
It compares to the information in `/hpc/group/cosmology/phy-lsst/work/dms118/roman_imsim/`
It writes output to hardcoded `/hpc/group/cosmology/lna18/roman_sim_imgs/Roman_Rubin_Sims_2024/` path with the SN ID.
