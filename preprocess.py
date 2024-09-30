# temp import for use with nsys
# (see "with nvtx.annotate" blocks below)
import nvtx

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
import time
import json
import numpy as np

# Make numpy stop thread hogging.
# I don't think I need this if I have it in the batch file, right?
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

# IMPORTS Astro:
from astropy.table import Table
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
from phrosty.utils import get_transient_info, set_logger, get_templates, get_science

###########################################################################
# Get environment variables.

infodir = os.getenv('SN_INFO_DIR', None)
assert infodir is not None, 'You need to set SN_INFO_DIR as an environment variable.'

sims_dir = os.getenv('SIMS_DIR', None)
assert sims_dir is not None, 'You need to set SIMS_DIR as an environment variable.'

dia_out_dir = os.getenv('DIA_OUT_DIR', None)
assert dia_out_dir is not None, 'You need to set DIA_INFO_DIR as an environment variable.'

###########################################################################

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
    output_path, skylvl, skyrms, DETECT_MASK = sky_subtract(band=band, pointing=pointing, sca=sca, force=True)
    return output_path, skyrms, DETECT_MASK

def preprocess(ra,dec,band,pair_info,
               skysub_dir=os.path.join(dia_out_dir,'skysub'),
               verbose=False,logger=None):

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
    t_skysub = os.path.join(skysub_dir,f'skysub_Roman_TDS_simple_model_{band}_{t_pointing}_{t_sca}.fits')

    logger.info( "*** Starting imalign" )
    t_align_savename = f'skysub_Roman_TDS_simple_model_{band}_{t_pointing}_{t_sca}_-_{band}_{sci_pointing}_{sci_sca}.fits'
    with nvtx.annotate( "imalign", color="yellow" ):
        t_align = imalign(template_path=sci_skysub_path,sci_path=t_skysub,savename=t_align_savename,force=True) # NOTE: This is correct, not flipped.

    logger.debug(f'Path to sky-subtracted science image: \n {sci_skysub_path}')
    logger.debug(f'Path to aligned, sky-subtracted template image: \n {t_align}')

    template_overlap = check_overlap(ra,dec,t_align,verbose=verbose)
    science_overlap = check_overlap(ra,dec,sci_skysub_path,verbose=verbose)

    if not template_overlap or not science_overlap:
        if verbose:
            logger.debug(f'{proc} Images {band} {sci_pointing} {sci_sca} and {band} {t_pointing} {t_sca} do not sufficiently overlap to do image subtraction.')

        return None, None

    else:
        # TODO : think about config file location and configurability / parameters
        logger.info( "*** Starting get_imsim_psf calls" )
        with nvtx.annotate( "get_imsim_psf", color="red" ):
            sci_psf_path = get_imsim_psf(ra, dec, band, sci_pointing, sci_sca,
                                         config_yaml_file=os.path.join( os.getenv("SN_INFO_DIR"), "tds.yaml" ) )
            if verbose:
                logger.debug(f'Path to science PSF: \n {sci_psf_path}')

            t_imsim_psf = get_imsim_psf(ra, dec, band, t_pointing, t_sca, logger=logger,
                                        config_yaml_file=os.path.join( os.getenv("SN_INFO_DIR"), "tds.yaml" ) )

        with nvtx.annotate( "rotate_psf", color="yellow" ):
            rot_psf_name = f'rot_psf_{band}_{t_pointing}_{t_sca}_-_{band}_{sci_pointing}_{sci_sca}.fits'
            t_psf_path = rotate_psf(ra,dec,t_imsim_psf,sci_skysub_path,savename=rot_psf_name,force=True)
            if verbose:
                logger.debug(f'Path to template PSF: \n {t_psf_path}')


        sci_conv_name = f'conv_sci_Roman_TDS_simple_model_{band}_{sci_pointing}_{sci_sca}_-_{band}_{t_pointing}_{t_sca}.fits'
        ref_conv_name = f'conv_ref_Roman_TDS_simple_model_{band}_{t_pointing}_{t_sca}_-_{band}_{sci_pointing}_{sci_sca}.fits'
        sci_conv, temp_conv = crossconvolve(sci_skysub_path,sci_psf_path,t_align,t_imsim_psf,sci_outname=sci_conv_name,ref_outname=ref_conv_name,force=True)
        if verbose:
            logger.debug(f'Path to cross-convolved science image: \n {sci_conv} \n Path to cross-convolved template image: \n {temp_conv}')

        #cuda.close()
            
def run(oid, band, sci_list_path, template_list_path, cpus_per_task=1, verbose=False):
# def run(oid, band, n_templates=1, cpus_per_task=1, verbose=False):
    
    ###################################################################
    # Start tracemalloc.
    tracemalloc.start()

    ###################################################################

    logger = set_logger( "preprocess_run", "pre_run" )
    
    if verbose:
        start_time = time.time()

    # logger.info( "***** Getting transient info" )
    # ra,dec,start,end = get_transient_info(oid)
    # logger.info( "***** Getting template_list" )
    # template_list = get_templates(oid,band,infodir,n_templates,verbose=verbose)
    # logger.info( "***** Getting science_list" )
    # science_list = get_science(oid,band,infodir,verbose=verbose)

    ra,dec,start,end = get_transient_info(oid)

    science_tab = Table.read(sci_list_path)
    science_tab = science_tab[science_tab['filter'] == band]
    science_list = [dict(zip(science_tab.colnames,row)) for row in science_tab]

    template_tab = Table.read(template_list_path)
    template_tab = template_tab[template_tab['filter'] == band]
    template_list = [dict(zip(template_tab.colnames,row)) for row in template_tab]
    
    pairs = list(itertools.product(template_list,science_list))

    logger.info( "***** Doing skysub" )
    with nvtx.annotate( "skysub", color="red" ):
         # First, unzip and sky subtract the images in their own multiprocessing pool.
        all_list = template_list + science_list
        all_list = list(dict(t) for t in {tuple(d.items) for d in all_list}) # Remove duplicates 
        for img in all_list:
            skysub_img_path, skyrms, DETECT_MASK = skysub(img)
            os.makedirs(os.path.join(dia_out_dir, 'skyrms'), exist_ok=True)
            skysub_img_basename = os.path.basename(skysub_img_path)
            skyrmspath = os.path.join(dia_out_dir, f'skyrms/{skysub_img_basename}.json')
            # TODO : worry about using skyrms.mean(), think if we should use something else
            json.dump( { 'skyrms': skyrms.mean() }, open( skyrmspath, "w" ) )
            os.makedirs( os.path.join( dia_out_dir, 'detect_mask' ), exist_ok=True )
            fname = os.path.join( dia_out_dir, f'detect_mask/{skysub_img_basename}.npy' )
            logger.info( f"Writing detection mask to {fname}" )
            np.save( fname, DETECT_MASK )
             

#        with Pool(cpus_per_task) as pool:
#            process = pool.map(skysub, all_list)
#            pool.close()
#            pool.join()


    if verbose:
        print('\n ******************************************************** \n Images have been sky-subtracted. \n  ******************************************************** \n')

#    partial_preprocess = partial(preprocess,ra,dec,band,verbose=verbose)

    logger.info( "***** Calling preprocess" )
    for pair in pairs:
        try:
            preprocess(ra,dec,band,pair,verbose=verbose)
        except Exception as exe:
            print(f'WARNING! EXCEPTION OCCURRED ON {pair}!')
            print(exe)
            print(' ******************************************************** \n')
#        with Manager() as mgr:
#            mgr_pairs = mgr.list(pairs)
#            with Pool(cpus_per_task) as pool_2:
#                process_2 = pool_2.map(partial_preprocess,mgr_pairs)
#                pool_2.close()
#                pool_2.join()

    if verbose:
        print('\n ******************************************************** \n Templates aligned, PSFs retrieved and aligned, images cross-convolved. \n  ******************************************************** \n')
        print(f'RUNTIMEPRINT preprocess.py: {time.time()-start_time}')

    ###################################################################
    # Print tracemalloc.
    current, peak = tracemalloc.get_traced_memory()
    print(f'MEMPRINT preprocess.py: Current memory = {current}, peak memory = {peak}')

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
        "--sci-list-path",
        type=str,
        help='Path to list of science images.'
    )

    parser.add_argument(
        "--template-list-path",
        type=str,
        help='Path to list of template images.'
    )

    # parser.add_argument(
    #     "--n-templates",
    #     type=int,
    #     help='Number of template images to use.'
    # )

    parser.add_argument(
        '--cpus-per-task',
        type=int,
        default=None,
        help="Number of CPUs for each multiprocessing pool task"
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

    cpus_per_task = args.cpus_per_task
    if cpus_per_task is None:
        # TODO : default when no slurm
        cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])

    # run(args.oid, args.band, args.n_templates, cpus_per_task=cpus_per_task, verbose=args.verbose)
    run(args.oid, args.band, args.sci_list_path, args.template_list_path, cpus_per_task=cpus_per_task, verbose=args.verbose)
    print("Finished with preprocess.py!")

if __name__ == '__main__':
    parse_and_run()
