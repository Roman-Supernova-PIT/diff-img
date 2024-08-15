# IMPORTS Standard:
import logging
# import tracemalloc
from multiprocessing import Pool, Manager, current_process
from functools import partial
import itertools
import argparse
import os
import sys
import re
import matplotlib.pyplot as plt
import time

# Make numpy stop thread hogging. 
# I don't think I need this if I have it in the batch file, right? 
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

# IMPORTS Astro:
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.nddata.utils import PartialOverlapError, NoOverlapError
from astropy.visualization import ZScaleInterval
import astropy.units as u

# IMPORTS Internal:
from phrosty.imagesubtraction import sky_subtract, imalign, get_imsim_psf, rotate_psf, crossconvolve
from phrosty.utils import get_transient_info, transient_in_or_out

###########################################################################
# Get environment variables. 

infodir = os.getenv('SN_INFO_DIR', None)
assert infodir is not None, 'You need to set SN_INFO_DIR as an environment variable.'

sims_dir = os.getenv('SIMS_DIR', None)
assert sims_dir is not None, 'You need to set SIMS_DIR as an environment variable.'

dia_out_dir = os.getenv('DIA_OUT_DIR', None)
assert infodir is not None, 'You need to set DIA_INFO_DIR as an environment variable.'

###########################################################################

def set_logger(proc,name):
    # Configure logger (Rob)
    logger = logging.getLogger(f'{name}_{proc}')
    if not logger.hasHandlers():
        log_out = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(f'[%(asctime)s - {proc} - %(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        log_out.setFormatter(formatter)
        logger.addHandler(log_out)
        logger.setLevel(logging.DEBUG) # ERROR, WARNING, INFO, or DEBUG (in that order by increasing detail)
    return logger

def get_templates(oid,band,n_templates=1,verbose=False):
    ra,dec,start,end = get_transient_info(oid)

    filepath = os.path.join(infodir,f'{oid}/{oid}_instances.csv')
    in_tab,out_tab = transient_in_or_out(oid,start,end,band,transient_info_filepath=filepath)

    template_tab = out_tab[:n_templates]
    if verbose:
        print('The template images are:')
        print(template_tab)

    template_list = [dict(zip(template_tab.colnames,row)) for row in template_tab]

    return template_list

def get_science(oid,band,verbose=False):
    ra,dec,start,end = get_transient_info(oid)

    filepath = os.path.join(infodir,f'{oid}/{oid}_instances.csv')
    in_tab,out_tab = transient_in_or_out(oid,start,end,band,transient_info_filepath=filepath)

    if verbose:
        print('The science images are:')
        print(in_tab)
        
    science_list = [dict(zip(in_tab.colnames,row)) for row in in_tab]
    
    return science_list

def check_overlap(ra,dec,imgpath,data_ext=0,overlap_size=500,verbose=False,show_cutout=False):
    """
    Does the science and template images sufficiently overlap, centered on
    the specified RA, dec coordinates (SN location)?
    """
    
    with fits.open(imgpath) as hdu:
        img = hdu[data_ext].data
        wcs = WCS(hdu[data_ext].header)
        
    coord = SkyCoord(ra=ra*u.deg,dec=dec*u.deg)
    pxcoords = skycoord_to_pixel(coord,wcs)

    try:
        cutout = Cutout2D(img,pxcoords,overlap_size,mode='strict')
        if show_cutout:
            z1,z2 = ZScaleInterval(n_samples=1000,
            contrast=0.25,
            max_reject=0.5,
            min_npixels=5,
            krej=2.5,
            max_iterations=5,
            ).get_limits(cutout.data)

            plt.figure()
            plt.imshow(cutout.data, vmin=z1, vmax=z2, cmap='Greys')

            plt.show()

        if verbose:
            print(f'The image {imgpath} contains sufficient overlap around coordinates {ra}, {dec}.')

        return True

    except (PartialOverlapError, NoOverlapError):
        if verbose:
            print(f'{imgpath} does not sufficiently overlap with the SN. ')
        return False

def skysub(infodict):
    """
    Wrapper for phrosty.imagesubtraction.sky_subtract() that allows input of
    a single dictionary for filter, pointing, and sca instead of individual
    values for each argument. Use this to enable use of multiprocessing.Pool.map.
    """
    band, pointing, sca = infodict['filter'], infodict['pointing'], infodict['sca']
    sky_subtract(band=band,pointing=pointing,sca=sca)

def preprocess(ra,dec,band,pair_info,
               skysub_dir=os.path.join(dia_out_dir,'skysub'),
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
    logger = set_logger(proc,'preprocess')

    ###########################################################################

    t_info, sci_info = pair_info
    sci_pointing, sci_sca = sci_info['pointing'], sci_info['sca']
    t_pointing, t_sca = t_info['pointing'], t_info['sca']

    sci_skysub_path = os.path.join(skysub_dir,f'skysub_Roman_TDS_simple_model_{band}_{sci_pointing}_{sci_sca}.fits')
    t_skysub = sci_skysub_path = os.path.join(skysub_dir,f'skysub_Roman_TDS_simple_model_{band}_{t_pointing}_{t_sca}.fits')

    t_align_savename = f'skysub_Roman_TDS_simple_model_{band}_{t_pointing}_{t_sca}_-_{band}_{sci_pointing}_{sci_sca}.fits'
    t_align = imalign(template_path=sci_skysub_path,sci_path=t_skysub,savename=t_align_savename,force=True) # NOTE: This is correct, not flipped.

    template_overlap = check_overlap(ra,dec,t_align,verbose=verbose)
    science_overlap = check_overlap(ra,dec,sci_skysub_path,verbose=verbose)

    if not template_overlap or not science_overlap:
        if verbose:
            logger.debug(f'{proc} Images {band} {sci_pointing} {sci_sca} and {band} {t_pointing} {t_sca} do not sufficiently overlap to do image subtraction.')

        return None, None

    else:
        sci_psf_path = get_imsim_psf(ra,dec,band,sci_pointing,sci_sca)
        if verbose:
            logger.debug(f'Path to science PSF: \n {sci_psf_path}')

        t_imsim_psf = get_imsim_psf(ra,dec,band,t_pointing,t_sca,logger=logger)

        rot_psf_name = f'rot_psf_{band}_{t_pointing}_{t_sca}_-_{band}_{sci_pointing}_{sci_sca}.fits'
        t_psf_path = rotate_psf(ra,dec,t_imsim_psf,sci_skysub_path,savename=rot_psf_name,force=True)
        if verbose:
            logger.debug(f'Path to template PSF: \n {t_psf_path}')

        sci_conv_name = f'conv_sci_Roman_TDS_simple_model_{band}_{t_pointing}_{t_sca}_-_{band}_{sci_pointing}_{sci_sca}.fits'
        ref_conv_name = f'conv_ref_Roman_TDS_simple_model_{band}_{t_pointing}_{t_sca}_-_{band}_{sci_pointing}_{sci_sca}.fits'
        sci_conv, temp_conv = crossconvolve(sci_skysub_path,sci_psf_path,t_align,t_imsim_psf,sci_outname=sci_conv_name,ref_outname=ref_conv_name,force=True)
        if verbose:
            logger.debug(f'Path to cross-convolved science image: \n {sci_conv} \n Path to cross-convolved template image: \n {temp_conv}')

def run(oid,band,n_templates=1,verbose=False):

    ###########################################################################

    # Get process name. 
    me = current_process()
    match = re.search( '([0-9]+)', me.name)
    if match is not None:
        proc = match.group(1)
    else:
        proc = str(me.pid)

    # Set logger.
    logger = set_logger(proc,'preprocess')

    ###########################################################################

    if verbose:
        start_time = time.time()

    ra,dec,start,end = get_transient_info(oid)
    template_list = get_templates(oid,band,n_templates,verbose=verbose)
    science_list = get_science(oid,band,verbose=verbose)
    pairs = list(itertools.product(template_list,science_list))

    # First, unzip and sky subtract the images in their own multiprocessing pool.
    all_list = template_list + science_list
    cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
    with Pool(cpus_per_task) as pool:
        process = pool.map(skysub, all_list)
        pool.close()
        pool.join()

    if verbose:
        logger.debug('\n ******************************************************** \n Images have been sky-subtracted. \n  ******************************************************** \n')

    partial_preprocess = partial(preprocess,ra,dec,band,verbose=verbose)

    with Manager() as mgr:
        mgr_pairs = mgr.list(pairs)
        with Pool(cpus_per_task) as pool_2:
            process_2 = pool_2.map(partial_preprocess,mgr_pairs)
            pool_2.close()
            pool_2.join()

    if verbose:
        logger.debug('\n ******************************************************** \n Templates aligned, PSFs retrieved and aligned, images cross-convolved. \n  ******************************************************** \n')
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
        prog='preprocess',
        description='Sky-subtracts, aligns, and cross-convolves images and their PSFs.'
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
    print("Finished with preprocess.py!")

if __name__ == '__main__':
    parse_and_run()