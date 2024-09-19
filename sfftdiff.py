import nvtx

# IMPORTS Standard:
import logging
import tracemalloc
import itertools
import argparse
import os
import sys
import re
import time
import cupy as cp

# Make numpy stop thread hogging. 
# I don't think I need this if I have it in the batch file, right? 
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import numpy as np

# IMPORTS Astro:
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
import astropy.units as u

# IMPORTS Internal:
from phrosty.imagesubtraction import difference
from phrosty.utils import get_transient_radec, get_transient_info, transient_in_or_out, set_logger, get_templates, get_science

###########################################################################
# Get environment variables. 

infodir = os.getenv('SN_INFO_DIR', None)
assert infodir is not None, 'You need to set SN_INFO_DIR as an environment variable.'

sims_dir = os.getenv('SIMS_DIR', None)
assert sims_dir is not None, 'You need to set SIMS_DIR as an environment variable.'

dia_out_dir = os.getenv('DIA_OUT_DIR', None)
assert infodir is not None, 'You need to set DIA_INFO_DIR as an environment variable.'

###########################################################################

def sfft(oid,band,
         sci_pointing,sci_sca,
         template_pointing,template_sca,
         verbose=False,
         backend='Cupy',
         cudadevice='0',
         nCPUthreads=1,
         logger=None):

    ra, dec = get_transient_radec(oid)

    # Path to convolved science image:
    sci_conv = os.path.join(dia_out_dir,f'convolved/conv_sci_Roman_TDS_simple_model_{band}_{sci_pointing}_{sci_sca}_-_{band}_{template_pointing}_{template_sca}.fits')

    # Path to convolved template image:
    template_conv = os.path.join(dia_out_dir,f'convolved/conv_ref_Roman_TDS_simple_model_{band}_{template_pointing}_{template_sca}_-_{band}_{sci_pointing}_{sci_sca}.fits')

    # Path to science PSF:
    sci_psf_path = os.path.join(dia_out_dir,f'psf/psf_{ra}_{dec}_{band}_{sci_pointing}_{sci_sca}.fits')

    # Path to template PSF:
    template_psf_path = os.path.join(dia_out_dir,f'psf/rot_psf_{band}_{template_pointing}_{template_sca}_-_{band}_{sci_pointing}_{sci_sca}.fits')

    paths = [sci_conv,template_conv,sci_psf_path,template_psf_path]
    pathexists = [os.path.exists(p) for p in paths]
    if all(pathexists):

        diff_savename = f'{band}_{sci_pointing}_{sci_sca}_-_{band}_{template_pointing}_{template_sca}.fits' # 'diff_' gets prepended to the beginning of this
        diff, soln = difference(sci_conv,template_conv,
                                savename=diff_savename,
                                backend=backend,cudadevice=cudadevice,
                                nCPUthreads=1,force=True,logger=logger)
        if verbose:
            logger.debug(f'Path to differenced image: \n {diff}')

        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()

        print(f'GPU MEMPRINT sfftdiff.sfft(): Memory pool used bytes = {mempool.used_bytes()}')
        print('Now freeing blocks.')

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    else:
        print(f'Preprocessed files for {band}_{sci_pointing}_{sci_sca}_-_{band}_{template_pointing}_{template_sca} do not exist. Skipping.')

def run(oid,band,n_templates=1,verbose=False):

    ###########################################################################

    # Set logger.
    logger = set_logger('SERIAL','sfftdiff')

    ###########################################################################

    # Start tracemalloc. 
    tracemalloc.start()

    ###########################################################################

    mempool_all = cp.get_default_memory_pool()

    if verbose:
        start_time = time.time()

    ra,dec,start,end = get_transient_info(oid)
    template_list = get_templates(oid,band,infodir,n_templates,verbose=verbose)
    science_list = get_science(oid,band,infodir,verbose=verbose)
    pairs = list(itertools.product(template_list,science_list))

    for pair in pairs:
        with nvtx.annotate( "sfft_diff_one_pair", color=0x88ff88 ):
            template_info = pair[0]
            sci_info = pair[1]

            template_pointing, template_sca = template_info['pointing'], template_info['sca']
            sci_pointing, sci_sca = sci_info['pointing'], sci_info['sca']

            sfft(oid,band,
                 sci_pointing,sci_sca,
                 template_pointing,template_sca,
                 verbose=verbose,logger=logger)

    if verbose:
        print(f'Difference imaging complete. \n RUNTIMEPRINT sfftdiff.py: {time.time()-start_time} \n GPU MEMPRINT sfftdiff.py {mempool_all.used_bytes()}')


    ###################################################################
    # Print tracemalloc.
    current, peak = tracemalloc.get_traced_memory()
    print(f'CPU MEMPRINT sfftdiff.py: Current memory = {current}, peak memory = {peak}')

    ###################################################################

def parse_slurm():
    """
    Turn a SLURM array ID into a band. 
    """
    sys.path.append(os.getcwd())
    taskID = int(os.environ["SLURM_ARRAY_TASK_ID"])

    print("taskID:", taskID)

    config = {1: "F184", 2: "H158", 3: "J129", 4: "K213", 5: "R062", 6: "Y106", 7: "Z087"}

    band = config[taskID]
    print('Band:', band)

    return band

def parse_and_run():
    parser = argparse.ArgumentParser(
        prog='sfft_and_animate', 
        description='Runs SFFT subtraction on images.'
    )

    parser.add_argument(
        'oid', 
        type=int, 
        help='ID of transient. Used to look up information on transient.'
    )

    parser.add_argument(
        "--band",
        type=str,
        default=None,
        choices=[None, "F184", "H158", "J129", "K213", "R062", "Y106", "Z087"],
        help="Filter to use.  None to use all available.  Overridden by --slurm_array.",
    )

    parser.add_argument(
        "--n-templates",
        type=int,
        help='Number of template images to use.'
    )

    parser.add_argument(
        '--verbose',
        type=bool,
        default=False,
        help='Talkativeness of code.'
    )

    parser.add_argument(
        "--slurm_array",
        default=False,
        action="store_true",
        help="If we're a slurm array job we're going to process the band_idx from the SLURM_ARRAY_TASK_ID.",
    )

    args = parser.parse_args()

    if args.slurm_array:
        args.band = parse_slurm()

    if args.band is None:
        print("Must specify either '--band' xor ('--slurm_array' and have SLURM_ARRAY_TASK_ID defined).")
        sys.exit()

    run(args.oid, args.band, args.n_templates, args.verbose)
    print("Finished with sfftdiff.py!")

if __name__ == '__main__':
    parse_and_run()
