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
import cupy as cp

from sfft.SpaceSFFTCupyFlow import SpaceSFFT_CupyFlow

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
from phrosty.utils import get_transient_info, set_logger, get_templates, get_science, _build_filepath

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

# def skysub(ra, dec, infodict, verbose=False):
#     """
#     Wrapper for phrosty.imagesubtraction.sky_subtract() that allows input of
#     a single dictionary for filter, pointing, and sca instead of individual
#     values for each argument. Use this to enable use of multiprocessing.Pool.map.
#     """
#     band, pointing, sca = infodict['filter'], infodict['pointing'], infodict['sca']
#     checkpath = _build_filepath(path=None,band=band,pointing=pointing,sca=sca,filetype='image',rootdir=sims_dir)
#     overlap = check_overlap(ra,dec,checkpath,verbose=verbose,data_ext=1)

#     if not overlap:
#         return None, None, None
#     else:
#         output_path, skylvl, skyrms, DETECT_MASK = sky_subtract(band=band, pointing=pointing, sca=sca, force=True)
#         return output_path, skyrms, DETECT_MASK

def get_transient_info_and_images( oid, band, sci_list_path, template_list_path ):
    ra, dec, start, end = get_transient_info(oid)
    objinfo = { 'oid': oid,
                'ra': ra,
                'dec': dec,
                'band': band,
                'start': start,
                'end': end }
    
    science_tab = Table.read(sci_list_path)
    science_tab = science_tab[science_tab['filter'] == band]
    science_imgs = { ( row['filter'], row['pointing'], row['sca'] ):
                     { 'filter': row['filter'],
                       'pointing': row['pointing'],
                       'sca': row['sca'],
                       'path': _build_filepath( path=None, band=row['filter'], pointing=row['pointing'], sca=row['sca'],
                                                filetype='image', rootdir=sims_dir ) }
                     for row in science_tab }
    # TODO : think about data_ext and what images are what and so forth
    science_imgs = { k: v for k, v in science_imgs.items() if check_overlap( ra, dec, v['path'], data_ext=1 ) }

    template_tab = Table.read(template_list_path)
    template_tab = template_tab[template_tab['filter'] == band]
    template_imgs = { ( row['filter'], row['pointing'], row['sca'] ):
                      { 'filter': row['filter'],
                        'pointing': row['pointing'],
                        'sca': row['sca'],
                        'path': _build_filepath( path=None, band=row['filter'], pointing=row['pointing'], sca=row['sca'],
                                                 filetype='image', rootdir=sims_dir ) }
                      for row in template_tab }
    template_imgs = { k: v for k, v in template_imgs.items() if check_overlap( ra, dec, v['path'], data_ext=1 ) }

    return objinfo, template_imgs, science_imgs

def run_get_imsim_psf( img, objinfo, config_yaml_file ):
    psf_path = get_imsim_psf( objinfo['ra'], objinfo['dec'], objinfo['band'],
                              img['pointing'], img['sca'], config_yaml_file=config_yaml_file )
    return psf_path

def save_psf_path( all_imgs, imgkey, psf_path ):
    print( f"Addfing psf_path to imgkey {imgkey}\n" )
    all_imgs[imgkey]['psf_path'] = psf_path

def get_psfs( objinfo, template_imgs, science_imgs, cpus_per_task=1,
              config_yaml_file=os.path.join( os.getenv("SN_INFO_DIR"), "tds.yaml" ) ):
    logger = set_logger( "preprocess_get_psfs", "pre_get_psfs" )

    ra = objinfo['ra']
    dec = objinfo['dec']
    band = objinfo['band']

    all_imgs = { k: v for k, v in zip( list( template_imgs.keys() ) + list( science_imgs.keys() ),
                                       list( template_imgs.values() ) + list( science_imgs.values() ) ) }

    if cpus_per_task > 1:
        with Pool( cpus_per_task ) as pool:
            for imgkey, img in all_imgs.items():
                callback_partial = partial( save_psf_path, all_imgs, imgkey )
                pool.apply_async( run_get_imsim_psf, (img, objinfo, config_yaml_file), {},
                                  callback_partial,
                                  lambda x: logger.error( f"get_imsim_psf subprocess failure: {x}" ) )
            pool.close()
            pool.join()
    else:
        for imgkey, img in all_imgs.items():
            save_psf_path( imgkey, run_get_imsim_psf( img ) )

    template_imgs = { k: all_imgs[k] for k in template_imgs.keys() }
    science_imgs = { k: all_imgs[k] for k in science_imgs.keys() }
    return template_imgs, science_imgs
                

def run_sky_subtract( img ):
    outpath, sky, skyrms, detect_mask = sky_subtract( path=img['path'], out_path=dia_out_dir, force=True )
    skyrms = np.median(skyrms)
    return ( skyrms, detect_mask )

def save_sky_subtract_result( all_imgs, imgkey, skyrms_and_detect_mask ):
    skyrms, detect_mask = skyrms_and_detect_mask
    all_imgs[imgkey]['skyrms'] = skyrms
    all_imgs[imgkey]['detect_mask'] = detect_mask

def sky_sub_all_images( template_imgs, science_imgs, cpus_per_task=1 ):
    all_imgs = { k: v for k, v in zip( list( template_imgs.keys() ) + list( science_imgs.keys() ),
                                       list( template_imgs.values() ) + list( science_imgs.values() ) ) }

    logger = set_logger( "preprocess_sky_sub", "pre_sky_sub" )

    if cpus_per_task > 1:
        with Pool( cpus_per_task ) as pool:
            for imgkey, img in all_imgs.items():
                callback_partial = partial( save_sky_subtract_result, all_imgs, imgkey )
                pool.apply_async( run_sky_subtract, (img,), {},
                                  callback=callback_partial,
                                  error_callback=lambda x: logger.error( f"Sky subtraction subprocess failure: {x}" ) )
            pool.close()
            pool.join()
    else:
        for imgkey, img in all_imgs.items():
            save_sky_subtract_result( imgkey, run_sky_subtract( img ) )

    template_imgs = { k: all_imgs[k] for k in template_imgs.keys() }
    science_imgs = { k: all_imgs[k] for k in science_imgs.keys() }
    return template_imgs, science_imgs

# def run(oid, band, sci_list_path, template_list_path, cpus_per_task=1, verbose=False):
# # def run(oid, band, n_templates=1, cpus_per_task=1, verbose=False):
    

#     ###################################################################
#     # Start tracemalloc.
#     tracemalloc.start()

#     ###################################################################

#     logger = set_logger( "preprocess_run", "pre_run" )
    
#     if verbose:
#         start_time = time.time()

#     pairs = get_pairs( oid, sci_list_path, template_list_path )
    
#     logger.info( "***** Doing skysub" )

#     with nvtx.annotate( "skysub", color="red" ):
#          # First, unzip and sky subtract the images in their own multiprocessing pool.
#         all_list = template_list + science_list
#         all_list = [dict(t) for t in {tuple(d.items()) for d in all_list}] # Remove duplicates
#         for img in all_list:
#             skysub_img_path, skyrms, DETECT_MASK = skysub(ra, dec, img, verbose)
#             img['skyrms'] = np.median( skyrms )
#             img['detect_mask'] = DETECT_MASK
#             outvars = [skysub_img_path, skyrms, DETECT_MASK]
#             if all(v is None for v in [skysub_img_path, skyrms, DETECT_MASK]):
#                 logger.info(f"There was not sufficient overlap with the SN for sky subtraction to be worth the time!")
#             else:
#                 os.makedirs(os.path.join(dia_out_dir, 'skyrms'), exist_ok=True)
#                 skysub_img_basename = os.path.basename(skysub_img_path)
#                 skyrmspath = os.path.join(dia_out_dir, f'skyrms/{skysub_img_basename}.json')
#                 # TODO : worry about using skyrms.mean(), think if we should use something else
#                 json.dump( { 'skyrms': skyrms.mean() }, open( skyrmspath, "w" ) )
#                 os.makedirs( os.path.join( dia_out_dir, 'detect_mask' ), exist_ok=True )
#                 fname = os.path.join( dia_out_dir, f'detect_mask/{skysub_img_basename}.npy' )
#                 logger.info( f"Writing detection mask to {fname}" )
#                 np.save( fname, DETECT_MASK )
             

#        with Pool(cpus_per_task) as pool:
#            process = pool.map(skysub, all_list)
#            pool.close()
#            pool.join()


#     if verbose:
#         print('\n ******************************************************** \n Images have been sky-subtracted. \n  ******************************************************** \n')

#    partial_preprocess = partial(preprocess,ra,dec,band,verbose=verbose)

#     logger.info( "***** Calling preprocess" )
#     for pair in pairs:
#         preprocess(ra, dec, band, pair, verbose=verbose)
#        with Manager() as mgr:
#            mgr_pairs = mgr.list(pairs)
#            with Pool(cpus_per_task) as pool_2:
#                process_2 = pool_2.map(partial_preprocess,mgr_pairs)
#                pool_2.close()
#                pool_2.join()
# 
#     if verbose:
#         print('\n ******************************************************** \n Templates aligned, PSFs retrieved and aligned, images cross-convolved. \n  ******************************************************** \n')
#         print(f'RUNTIMEPRINT preprocess.py: {time.time()-start_time}')
# 
#     ###################################################################
#     # Print tracemalloc.
#     current, peak = tracemalloc.get_traced_memory()
#     print(f'MEMPRINT preprocess.py: Current memory = {current}, peak memory = {peak}')
# 
#     ###################################################################

def parse_slurm():
    """
    Turn a SLURM array ID into a band.
    """
    raise RuntimeError( "THIS IS BROKEN" )

    sys.path.append(os.getcwd())
    taskID = int(os.environ["SLURM_ARRAY_TASK_ID"])

    print("taskID:", taskID)

    config = {1: "F184", 2: "H158", 3: "J129", 4: "K213", 5: "R062", 6: "Y106", 7: "Z087"}

    band = config[taskID]
    print('Band:', band)

    return band

def parse_and_run():
    raise RuntimeError( "THIS IS BROKEN" )

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
