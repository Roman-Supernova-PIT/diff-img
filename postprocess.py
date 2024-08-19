# IMPORTS Standard:
import logging
import tracemalloc
from multiprocessing import Pool, Manager, current_process
from functools import partial
import itertools
import argparse
import os
import sys
import re
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time

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
from astropy.visualization import ZScaleInterval
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
import astropy.units as u

# IMPORTS Internal:
from phrosty.imagesubtraction import decorr_kernel, decorr_img, stampmaker
from phrosty.utils import get_transient_info, transient_in_or_out, set_logger, get_templates, get_science

###########################################################################
# Get environment variables. 

infodir = os.getenv('SN_INFO_DIR', None)
assert infodir is not None, 'You need to set SN_INFO_DIR as an environment variable.'

sims_dir = os.getenv('SIMS_DIR', None)
assert sims_dir is not None, 'You need to set SIMS_DIR as an environment variable.'

dia_out_dir = os.getenv('DIA_OUT_DIR', None)
assert dia_out_dir is not None, 'You need to set DIA_INFO_DIR as an environment variable.'

###########################################################################

def postprocess(ra,dec,band,pair_info,
                verbose=False):

    ###########################################################################

    # Get process name. 
    me = current_process()
    match = re.search( '([0-9]+)', me.name)
    if match is not None:
        proc = match.group(1)
    else:
        proc = str(me.pid)

    # Set logger.
    logger = set_logger(proc,'postprocess')

    ###########################################################################

    template_info, sci_info = pair_info
    sci_pointing, sci_sca = sci_info['pointing'], sci_info['sca']
    template_pointing, template_sca = template_info['pointing'], template_info['sca']


    # Set up input filepaths for generating the decorrelation kernel.
    skysub_dir = os.path.join(dia_out_dir,'skysub')
    sci_skysub_path = os.path.join(skysub_dir,f'skysub_Roman_TDS_simple_model_{band}_{sci_pointing}_{sci_sca}.fits')
    template_skysub_path = os.path.join(skysub_dir,f'skysub_Roman_TDS_simple_model_{band}_{template_pointing}_{template_sca}.fits')
    
    psf_dir = os.path.join(dia_out_dir,'psf')
    sci_psf_path = os.path.join(psf_dir,f'psf_{ra}_{dec}_{band}_{sci_pointing}_{sci_sca}.fits')
    template_psf_path = os.path.join(psf_dir,f'rot_psf_{band}_{template_pointing}_{template_sca}_-_{band}_{sci_pointing}_{sci_sca}.fits')
    
    subtract_dir = os.path.join(dia_out_dir,'subtract')
    diff_dir = os.path.join(subtract_dir,'difference')
    soln_dir = os.path.join(subtract_dir,'solution')
    solnpath = os.path.join(soln_dir,f'solution_{band}_{sci_pointing}_{sci_sca}_-_{band}_{template_pointing}_{template_sca}.fits')
    diffpath = os.path.join(diff_dir,f'diff_{band}_{sci_pointing}_{sci_sca}_-_{band}_{template_pointing}_{template_sca}.fits')

    dcker_savename = f'dcker_{band}_{sci_pointing}_{sci_sca}_-_{template_pointing}_{template_sca}.fits'

    # Generate decorrelation kernel.
    dcker_path = decorr_kernel(sci_skysub_path,template_skysub_path,
                               sci_psf_path,template_psf_path,
                               diffpath,solnpath,
                               savename=dcker_savename)
    if verbose:
        logger.debug(f'Path to decorrelation kernel: \n {dcker_path}')

    s5, p5 = tracemalloc.get_traced_memory()
    logger.debug(f'MEMORY AFTER GENERATING DCKER mem size = {s5}, mem peak = {p5}')
    tracemalloc.reset_peak()

    # Apply decorrelation kernel to images    
    decorr_imgpath = decorr_img(diffpath,dcker_path)
    if verbose:
        logger.debug(f'Path to final decorrelated differenced image: \n {decorr_imgpath}')

    sci_conv = os.path.join(dia_out_dir,f'convolved/conv_sci_Roman_TDS_simple_model_{band}_{template_pointing}_{template_sca}_-_{band}_{sci_pointing}_{sci_sca}.fits')
    zpt_savename = f'zptimg_{band}_{sci_pointing}_{sci_sca}_-_{template_pointing}_{template_sca}.fits'
    zpt_imgpath = decorr_img(sci_conv,dcker_path,savename=zpt_savename)
    if verbose:
        logger.debug(f'Path to zeropoint image: \n {zpt_imgpath}')

    # Apply decorrelation kernel to PSF
    decorr_psf_savename = f'psf_{band}_{sci_pointing}_{sci_sca}_-_{template_pointing}_{template_sca}.fits'
    decorr_psfpath = decorr_img(sci_psf_path,dcker_path,savename=decorr_psf_savename)
    if verbose:
        logger.debug(f'Path to decorrelated PSF (use for photometry): \n {decorr_psfpath}')

    s6, p6 = tracemalloc.get_traced_memory()
    logger.debug(f'MEMORY AFTER DECORRELATING THE IMAGES mem size = {s6}, mem peak = {p6}')
    tracemalloc.reset_peak()        

    # Make stamps
    skysub_stamp_savename = f'stamp_{ra}_{dec}_skysub_Roman_TDS_simple_model_{band}_{sci_pointing}_{sci_sca}.fits'
    skysub_stamp_path = stampmaker(ra,dec,sci_skysub_path,savename=skysub_stamp_savename,shape=np.array([100,100]))
    if verbose:
        logger.debug(f'Path to sky-subtracted-only SN stamp: \n {skysub_stamp_path}')

    dd_stamp_savename = f'stamp_{ra}_{dec}_diff_Roman_TDS_simple_model_{band}_{sci_pointing}_{sci_sca}.fits'
    dd_stamp_path = stampmaker(ra,dec,decorr_imgpath,savename=dd_stamp_savename,shape=np.array([100,100]))
    if verbose:
        print(f'Path to final decorrelated differenced SN stamp: \n {dd_stamp_path}')

    s7, p7 = tracemalloc.get_traced_memory()
    logger.debug(f'MEMORY AFTER MAKING STAMPS mem size = {s7}, mem peak = {p7}')
    tracemalloc.reset_peak()        

def run(oid,band,n_templates=1,verbose=False):

    if verbose:
        start_time = time.time()

    ra,dec,start,end = get_transient_info(oid)
    template_list = get_templates(oid,band,infodir,n_templates,verbose=verbose)
    science_list = get_science(oid,band,infodir,verbose=verbose)
    pairs = list(itertools.product(template_list,science_list))

    partial_postprocess = partial(postprocess,ra,dec,band,verbose=verbose)
    cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])

    with Manager() as mgr:
        mgr_pairs = mgr.list(pairs)
        with Pool(cpus_per_task) as pool:
            process = pool.map(partial_postprocess,mgr_pairs)
            pool.close()
            pool.join()

    if verbose:
        logger.debug('\n ******************************************************** \n Decorrelation kernel generated, applied to images and PSFs, and stamps generated. \n  ******************************************************** \n')
        logger.debug(f'Run time: {time.time()-start_time}')

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
        prog='postprocess',
        description='Generates decorrelation kernels, applies them to PSFs and images, generates stamps.'
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
    print("Finished with postprocess.py!")

if __name__ == '__main__':
    parse_and_run()