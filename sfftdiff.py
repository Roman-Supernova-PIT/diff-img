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
from phrosty.utils import ( get_transient_radec, get_transient_info, transient_in_or_out,
                            set_logger, get_templates, get_science, _build_filepath )

from sfft.SpaceSFFTCupyFlow import SpaceSFFT_CupyFlow

###########################################################################
# Get environment variables. 

infodir = os.getenv('SN_INFO_DIR', None)
assert infodir is not None, 'You need to set SN_INFO_DIR as an environment variable.'

sims_dir = os.getenv('SIMS_DIR', None)
assert sims_dir is not None, 'You need to set SIMS_DIR as an environment variable.'

dia_out_dir = os.getenv('DIA_OUT_DIR', None)
assert infodir is not None, 'You need to set DIA_INFO_DIR as an environment variable.'

###########################################################################

def align_and_pre_convolve(ra, dec, band, pair_info,
                           skysub_dir=os.path.join(dia_out_dir,'skysub'),
                           verbose=False,logger=None):

    # Set logger.
    logger = set_logger("",'preprocess')

    t_info, sci_info = pair_info
    sci_pointing, sci_sca = sci_info['pointing'], sci_info['sca']
    t_pointing, t_sca = t_info['pointing'], t_info['sca']
    sci_psf_path = sci_info['psf_path']
    templ_psf_path = t_info['psf_path']
    sci_detmask_path = sci_info['detect_mask']
    templ_detmask_path = t_info['detect_mask']
    sci_skyrms = sci_info['skyrms']
    templ_skyrms = sci_info['skyrms']
    
    orig_scipath = _build_filepath(path=None,band=band,pointing=sci_pointing,sca=sci_sca,filetype='image',rootdir=sims_dir)
    orig_tpath = _build_filepath(path=None,band=band,pointing=t_pointing,sca=t_sca,filetype='image',rootdir=sims_dir)

    sci_skysub_path = os.path.join(skysub_dir,f'skysub_Roman_TDS_simple_model_{band}_{sci_pointing}_{sci_sca}.fits')
    templ_skysub_path = os.path.join(skysub_dir,f'skysub_Roman_TDS_simple_model_{band}_{t_pointing}_{t_sca}.fits')

    logger.debug(f'Path to sky-subtracted science image: \n {sci_skysub_path}')
    logger.debug(f'Path to sky-subtracted template image: \n {templ_skysub_path}')

    with fits.open( sci_skysub_path ) as hdul:
        hdr_sci = hdul[0].header
        data_sci = cp.array( np.ascontiguousarray(hdul[0].data.T), dtype=cp.float64 )
    with fits.open( templ_skysub_path ) as hdul:
        hdr_templ = hdul[0].header
        data_templ = cp.array( np.ascontiguousarray(hdul[0].data.T), dtype=cp.float64 )

    with fits.open( sci_psf_path ) as hdul:
        sci_psf = cp.array( np.ascontiguousarray( hdul[0].data.T ), dtype=cp.float64 )

    with fits.open( templ_psf_path ) as hdul:
        templ_psf = cp.array( np.ascontiguousarray( hdul[0].data.T ), dtype=cp.float64 )

    with fits.open( sci_detmask_path ) as hdul:
        sci_detmask = cp.array( np.ascontiguousarray( hdul[0].data.T ) )
        
    with fits.open( templ_detmask_path ) as hdul:
        templ_detmask = cp.array( np.ascontiguousarray( hdul[0].data.T ) )

    sfftifier = SpaceSFFT_CupyFlow(
        hdr_sci, hdr_templ,
        sci_skyrms, templ_skyrms,
        data_sci, data_templ,
        sci_detmask, templ_detmask,
        sci_psf, templ_psf
    )
    sfftifier.resampling_image_mask_psf()
    sfftifier.cross_convolution()

    return sfftifier

def run(oid,band,sci_list_path,template_list_path,verbose=False):

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

    science_tab = Table.read(sci_list_path)
    science_tab = science_tab[science_tab['filter'] == band]
    science_list = [dict(zip(science_tab.colnames,row)) for row in science_tab]

    template_tab = Table.read(template_list_path)
    template_tab = template_tab[template_tab['filter'] == band]
    template_list = [dict(zip(template_tab.colnames,row)) for row in template_tab]

    pairs = list(itertools.product(template_list,science_list))

    for pair in pairs:
        with nvtx.annotate( "sfft_diff_one_pair", color=0x88ff88 ):
            template_info = pair[0]
            sci_info = pair[1]

            template_pointing, template_sca = template_info['pointing'], template_info['sca']
            sci_pointing, sci_sca = sci_info['pointing'], sci_info['sca']

            try:
                sfft(oid,band,
                    sci_pointing,sci_sca,
                    template_pointing,template_sca,
                    verbose=verbose,logger=logger)
            except Exception as exe:
                print(f'WARNING! EXCEPTION OCCURRED ON {pair}!')
                print(exe)
                print(' ******************************************************** \n')

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
    raise RuntimeError( "OMG BROKEN" )
    
    sys.path.append(os.getcwd())
    taskID = int(os.environ["SLURM_ARRAY_TASK_ID"])

    print("taskID:", taskID)

    config = {1: "F184", 2: "H158", 3: "J129", 4: "K213", 5: "R062", 6: "Y106", 7: "Z087"}

    band = config[taskID]
    print('Band:', band)

    return band

def parse_and_run():
    raise RuntimeError( "OMG BROKEN" )

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
        "--sci-list-path",
        type=str,
        help='Path to list of science images.'
    )

    parser.add_argument(
        "--template-list-path",
        type=str,
        help='Path to list of template images.'
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

    run(args.oid, args.band, args.sci_list_path, args.template_list_path, args.verbose)
    print("Finished with sfftdiff.py!")

if __name__ == '__main__':
    parse_and_run()
