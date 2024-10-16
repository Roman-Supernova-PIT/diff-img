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
from astropy.io import fits

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

    with nvtx.annotate( "template_info", color=0x8800ff ):
        template_info, sci_info = pair_info
        sci_pointing, sci_sca = sci_info['pointing'], sci_info['sca']
        template_pointing, template_sca = template_info['pointing'], template_info['sca']

    # First, check that the difference image exists in the first place.
    subtract_dir = os.path.join(dia_out_dir,'subtract')
    diff_dir = os.path.join(subtract_dir,'difference')
    diffpath = os.path.join(diff_dir,f'diff_{band}_{sci_pointing}_{sci_sca}_-_{band}_{template_pointing}_{template_sca}.fits')
    
    if os.path.exists(diffpath):
        # Check that you actually sky-subtracted and have the json files. 
        # However, if we have the difference image, we should have these,
        # and if we don't have the difference image, we shouldn't have these. 
        # But worry about this later. 

        template_skyrmspath = os.path.join(dia_out_dir,f'skyrms/skysub_Roman_TDS_simple_model_{band}_{template_pointing}_{template_sca}.fits.json')
        science_skyrmspath = os.path.join(dia_out_dir,f'skyrms/skysub_Roman_TDS_simple_model_{band}_{sci_pointing}_{sci_sca}.fits.json')

        if os.path.exists(template_skyrmspath) and os.path.exists(science_skyrmspath):

            # Set up input filepaths for generating the decorrelation kernel.
            skysub_dir = os.path.join(dia_out_dir,'skysub')
            sci_skysub_path = os.path.join(skysub_dir,f'skysub_Roman_TDS_simple_model_{band}_{sci_pointing}_{sci_sca}.fits')
            template_skysub_path = os.path.join(skysub_dir,f'skysub_Roman_TDS_simple_model_{band}_{template_pointing}_{template_sca}.fits')

            psf_dir = os.path.join(dia_out_dir,'psf')
            sci_psf_path = os.path.join(psf_dir,f'psf_{ra}_{dec}_{band}_{sci_pointing}_{sci_sca}.fits')
            template_psf_path = os.path.join(psf_dir,f'rot_psf_{band}_{template_pointing}_{template_sca}_-_{band}_{sci_pointing}_{sci_sca}.fits')

            soln_dir = os.path.join(subtract_dir,'solution')
            solnpath = os.path.join(soln_dir,f'solution_{band}_{sci_pointing}_{sci_sca}_-_{band}_{template_pointing}_{template_sca}.fits')
            diffpath = os.path.join(diff_dir,f'diff_{band}_{sci_pointing}_{sci_sca}_-_{band}_{template_pointing}_{template_sca}.fits')

            dcker_savename = f'dcker_{band}_{sci_pointing}_{sci_sca}_-_{band}_{template_pointing}_{template_sca}.fits'

            # Generate decorrelation kernel.
            with nvtx.annotate( "decorr_kernel", color="green" ):
                dcker_path = decorr_kernel(sci_skysub_path,template_skysub_path,
                                        sci_psf_path,template_psf_path,
                                        diffpath,solnpath,
                                        savename=dcker_savename)
            if verbose:
                logger.debug(f'Path to decorrelation kernel: \n {dcker_path}')

            # Apply decorrelation kernel to images
            with nvtx.annotate( "decorr_img diff", color="green" ):
                decorr_imgpath = decorr_img(diffpath,dcker_path)
            if verbose:
                logger.debug(f'Path to final decorrelated differenced image: \n {decorr_imgpath}')

            sci_conv = os.path.join(dia_out_dir,f'convolved/conv_sci_Roman_TDS_simple_model_{band}_{sci_pointing}_{sci_sca}_-_{band}_{template_pointing}_{template_sca}.fits')
            zpt_savename = f'zptimg_{band}_{sci_pointing}_{sci_sca}_-_{band}_{template_pointing}_{template_sca}.fits'
            with nvtx.annotate( "decorr_img zpt", color="green" ):
                zpt_imgpath = decorr_img(sci_conv,dcker_path,savename=zpt_savename)
            if verbose:
                logger.debug(f'Path to zeropoint image: \n {zpt_imgpath}')

            # Apply decorrelation kernel to PSF
            decorr_psf_savename = f'psf_{band}_{sci_pointing}_{sci_sca}_-_{band}_{template_pointing}_{template_sca}.fits'
            with nvtx.annotate( "decorr_img psf", color=0xaa44ff ):
                decorr_psfpath = decorr_img(sci_psf_path,dcker_path,savename=decorr_psf_savename)
            if verbose:
                logger.debug(f'Path to decorrelated PSF (use for photometry): \n {decorr_psfpath}')

            with nvtx.annotate( "stamp making", color="red" ):
                # Make stamps
                skysub_stamp_savename = f'stamp_{ra}_{dec}_skysub_Roman_TDS_simple_model_{band}_{sci_pointing}_{sci_sca}.fits'
                skysub_stamp_path = stampmaker(ra,dec,sci_skysub_path,savename=skysub_stamp_savename,shape=np.array([100,100]))
                if verbose:
                    logger.debug(f'Path to sky-subtracted-only SN stamp: \n {skysub_stamp_path}')

                decorr_zptimg_savename = f'stamp_{ra}_{dec}_decorr_zptimg_Roman_TDS_simple_model_{band}_{sci_pointing}_{sci_sca}.fits'
                decorr_zptimg_stamp = stampmaker(ra,dec,zpt_imgpath,savename=decorr_zptimg_savename,shape=np.array([100,100]))
                if verbose:
                    logger.debug(f'Path to sky-subtracted-only SN stamp: \n {skysub_stamp_path}')

                d_stamp_savename = f'stamp_{ra}_{dec}_decorr_diff_Roman_TDS_simple_model_{band}_{sci_pointing}_{sci_sca}_-_{band}_{template_pointing}_{template_sca}.fits'
                d_stamp_path = stampmaker(ra,dec,diffpath,savename=d_stamp_savename,shape=np.array([100,100]))
                if verbose:
                    logger.debug(f'Path to differenced SN stamp: \n {d_stamp_path}')

                dd_stamp_savename = f'stamp_{ra}_{dec}_decorr_diff_Roman_TDS_simple_model_{band}_{sci_pointing}_{sci_sca}_-_{band}_{template_pointing}_{template_sca}.fits'
                dd_stamp_path = stampmaker(ra,dec,decorr_imgpath,savename=dd_stamp_savename,shape=np.array([100,100]))
            if verbose:
                logger.debug(f'Path to final decorrelated differenced SN stamp: \n {dd_stamp_path}')
        else:
            print(f'Sky RMS jsons do not exist for either {band}_{sci_pointing}_{sci_sca} or {band}_{template_pointing}_{template_sca}! Skipping.')
    else:
        print(f'Difference imaging files for {band}_{sci_pointing}_{sci_sca}_-_{band}_{template_pointing}_{template_sca} do not exist. Skipping.')

# def run(oid, band, n_templates=1, cpus_per_task=1, verbose=False):
def run(oid, band, sci_list_path, template_list_path, cpus_per_task=1, verbose=False):

    ###########################################################################

    # Start tracemalloc.
    tracemalloc.start()

    ###########################################################################

    if verbose:
        start_time = time.time()

    print('postprocess started!')

    ra,dec,start,end = get_transient_info(oid)
    # template_list = get_templates(oid,band,infodir,n_templates,verbose=verbose)
    # science_list = get_science(oid,band,infodir,verbose=verbose)

    science_tab = Table.read(sci_list_path)
    science_tab = science_tab[science_tab['filter'] == band]
    science_list = [dict(zip(science_tab.colnames,row)) for row in science_tab]

    template_tab = Table.read(template_list_path)
    template_tab = template_tab[template_tab['filter'] == band]
    template_list = [dict(zip(template_tab.colnames,row)) for row in template_tab]

    pairs = list(itertools.product(template_list,science_list))

    partial_postprocess = partial(postprocess,ra,dec,band,verbose=verbose)

    for pair in pairs:
        partial_postprocess(pair)
#    with Manager() as mgr:
#        mgr_pairs = mgr.list(pairs)
#        with Pool(cpus_per_task) as pool:
#            process = pool.map(partial_postprocess,mgr_pairs)
#            pool.close()
#            pool.join()

    if verbose:
        print('\n ******************************************************** \n Decorrelation kernel generated, applied to images and PSFs, and stamps generated. \n  ******************************************************** \n')
        print(f'RUNTIMEPRINT postprocess.py: {time.time()-start_time}')

    ###################################################################
    # Print tracemalloc.
    current, peak = tracemalloc.get_traced_memory()
    print(f'MEMPRINT postprocess.py: Current memory = {current}, peak memory = {peak}')

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

    # run(args.oid, args.band, n_templates=args.n_templates, cpus_per_task=cpus_per_task, verbose=args.verbose)
    run(args.oid, args.band, args.sci_list_path, args.template_list_path, cpus_per_task=cpus_per_task, verbose=args.verbose)
    print("Finished with postprocess.py!")

if __name__ == '__main__':
    parse_and_run()
