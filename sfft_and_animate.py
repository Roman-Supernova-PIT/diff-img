# IMPORTS Standard:
import dask.bag as db
import logging
import tracemalloc
from multiprocessing import Pool, Manager, current_process
from multiprocessing.managers import SharedMemoryManager
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
from astropy.nddata.utils import PartialOverlapError, NoOverlapError
import astropy.units as u

# IMPORTS Internal:
from phrosty.imagesubtraction import sky_subtract, imalign, get_imsim_psf, rotate_psf, crossconvolve, difference, decorr_kernel, decorr_img, stampmaker
from phrosty.plotting import animate_stamps
from phrosty.utils import get_transient_radec, get_transient_mjd, get_mjd_info, _build_filepath

###########################################################################

def get_sn_info(oid):
    """
    Retrieve RA, Dec, MJD start, MJD end for specified object ID.  
    """
    RA, DEC = get_transient_radec(oid)
    start, end = get_transient_mjd(oid)

    return RA, DEC, start, end

def sn_in_or_out(oid,start,end,band,infodir='/hpc/group/cosmology/lna18/'):
    """
    Retrieve pointings that contain and do not contain the specified SN,
    per the truth files by MJD. 

    Returns a tuple of astropy tables (images with the SN, images without the SN).
    """
    transient_info_file = os.path.join(
        infodir, f'roman_sim_imgs/Roman_Rubin_Sims_2024/{oid}/{oid}_instances_nomjd.csv'
    )

    tab = Table.read(transient_info_file)
    tab.sort('pointing')
    tab = tab[tab['filter'] == band]

    in_all = get_mjd_info(start,end)
    in_rows = np.where(np.isin(tab['pointing'],in_all['pointing']))[0]
    in_tab = tab[in_rows]

    out_all = get_mjd_info(start,end,return_inverse=True)
    out_rows = np.where(np.isin(tab['pointing'],out_all['pointing']))[0]
    out_tab = tab[out_rows]

    return in_tab, out_tab

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

def sfft(ra,dec,band,pair_info,
                     skysub_dir='/work/lna18/imsub_out/skysub',
                     verbose=False):
    """
    Run SFFT steps. This includes:
    1. Aligning template image to science image
    2. Preprocessing the science image
    3. Cross-convolution
    4. Differencing
    5. Generating decorrelation kernel
    6. Applying decorrelation kernel to difference image and cross-convolved science image

    """

    # Get process name. 
    me = current_process()
    match = re.search( '([0-9]+)', me.name)
    if match is not None:
        proc = match.group(1)
    else:
        proc = str(me.pid)

    # Configure logger (Rob)
    logger = logging.getLogger(f'sfft_{proc}')
    if not logger.hasHandlers():
        log_out = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(f'[%(asctime)s - {proc} - %(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        log_out.setFormatter(formatter)
        logger.addHandler(log_out)
        logger.setLevel(logging.DEBUG) # ERROR, WARNING, INFO, or DEBUG (in that order by increasing detail)

    logger.debug('~~~~~~~~~~~~~~~~STARTING SFFT!~~~~~~~~~~~~~~~~')

    tracemalloc.start()
    t_info, sci_info = pair_info
    sci_pointing, sci_sca = sci_info['pointing'], sci_info['sca']
    t_pointing, t_sca = t_info['pointing'], t_info['sca']

    logger.debug(f' ********************************************************'
                 f'\n'
                 f'Template image: {band} {t_pointing} {t_sca} \n'
                 f'Science image: {band} {sci_pointing} {sci_sca} \n ********************************************************')
    
    sci_skysub_path = os.path.join(skysub_dir,f'skysub_Roman_TDS_simple_model_{band}_{sci_pointing}_{sci_sca}.fits')
    t_skysub = sci_skysub_path = os.path.join(skysub_dir,f'skysub_Roman_TDS_simple_model_{band}_{t_pointing}_{t_sca}.fits')

    t_align_savename = f'skysub_Roman_TDS_simple_model_{band}_{t_pointing}_{t_sca}_-_{band}_{sci_pointing}_{sci_sca}.fits'
    t_align = imalign(template_path=sci_skysub_path,sci_path=t_skysub,savename=t_align_savename,force=True) # NOTE: This is correct, not flipped.

    template_overlap = check_overlap(ra,dec,t_align,verbose=verbose)
    science_overlap = check_overlap(ra,dec,sci_skysub_path,verbose=verbose)
    
    s1, p1 = tracemalloc.get_traced_memory()
    logger.debug(f'MEMORY AFTER SKYSUB AND ALIGN mem size = {s1}, mem peak = {p1}')
    tracemalloc.reset_peak()

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

        s2, p2 = tracemalloc.get_traced_memory()
        logger.debug(f'MEMORY AFTER PSF STUFF mem size = {s2}, mem peak = {p2}')
        tracemalloc.reset_peak()

        sci_conv_name = f'conv_sci_Roman_TDS_simple_model_{band}_{t_pointing}_{t_sca}_-_{band}_{sci_pointing}_{sci_sca}.fits'
        ref_conv_name = f'conv_ref_Roman_TDS_simple_model_{band}_{t_pointing}_{t_sca}_-_{band}_{sci_pointing}_{sci_sca}.fits'
        sci_conv, temp_conv = crossconvolve(sci_skysub_path,sci_psf_path,t_align,t_imsim_psf,sci_outname=sci_conv_name,ref_outname=ref_conv_name,force=True)
        if verbose:
            logger.debug(f'Path to cross-convolved science image: \n {sci_conv} \n Path to cross-convolved template image: \n {temp_conv}')

        s3, p3 = tracemalloc.get_traced_memory()
        logger.debug(f'MEMORY AFTER CONVOLUTION mem size = {s3}, mem peak = {p3}')
        tracemalloc.reset_peak()

        diff_savename = f'{band}_{sci_pointing}_{sci_sca}_-_{t_pointing}_{t_sca}.fits' # 'diff_' gets appended to the beginning of this
        diff, soln = difference(sci_conv,temp_conv,sci_psf_path,t_psf_path,savename=diff_savename,backend='Numpy',nCPUthreads=1,force=True,logger=logger)
        if verbose:
            logger.debug(f'Path to differenced image: \n {diff}')

        s4, p4 = tracemalloc.get_traced_memory()
        logger.debug(f'MEMORY AFTER DIFFERENCING mem size = {s4}, mem peak = {p4}')
        tracemalloc.reset_peak()
        
        dcker_savename = f'dcker_{band}_{sci_pointing}_{sci_sca}_-_{t_pointing}_{t_sca}.fits'
        dcker_path = decorr_kernel(sci_skysub_path,t_skysub,sci_psf_path,t_psf_path,diff,soln,savename=dcker_savename)
        if verbose:
            logger.debug(f'Path to decorrelation kernel: \n {dcker_path}')

        s5, p5 = tracemalloc.get_traced_memory()
        logger.debug(f'MEMORY AFTER GENERATING DCKER mem size = {s5}, mem peak = {p5}')
        tracemalloc.reset_peak()
        
        decorr_imgpath = decorr_img(diff,dcker_path)
        if verbose:
            logger.debug(f'Path to final decorrelated differenced image: \n {decorr_imgpath}')

        zpt_savename = f'zptimg_{band}_{sci_pointing}_{sci_sca}_-_{t_pointing}_{t_sca}.fits'
        zpt_imgpath = decorr_img(sci_conv,dcker_path,savename=zpt_savename)
        if verbose:
            logger.debug(f'Path to zeropoint image: \n {zpt_imgpath}')

        decorr_psf_savename = f'psf_{band}_{sci_pointing}_{sci_sca}_-_{t_pointing}_{t_sca}.fits'
        decorr_psfpath = decorr_img(sci_psf_path,dcker_path,savename=decorr_psf_savename)
        if verbose:
            logger.debug(f'Path to decorrelated PSF (use for photometry): \n {decorr_psfpath}')

        s6, p6 = tracemalloc.get_traced_memory()
        logger.debug(f'MEMORY AFTER DECORRELATING THE IMAGES mem size = {s6}, mem peak = {p6}')
        tracemalloc.reset_peak()        

        skysub_stamp_savepath = '/work/lna18/imsub_out/skysub/stamps/'
        skysub_stamp_path = stampmaker(ra,dec,sci_skysub_path,savedir=skysub_stamp_savepath,shape=np.array([100,100]))
        if verbose:
            logger.debug(f'Path to sky-subtracted-only SN stamp: \n {skysub_stamp_path}')

        dd_stamp_savepath = '/work/lna18/imsub_out/decorr/stamps/'
        dd_stamp_path = stampmaker(ra,dec,decorr_imgpath,savedir=dd_stamp_savepath,shape=np.array([100,100]))
        if verbose:
            print(f'Path to final decorrelated differenced SN stamp: \n {dd_stamp_path}')

        s7, p7 = tracemalloc.get_traced_memory()
        logger.debug(f'MEMORY AFTER MAKING STAMPS mem size = {s7}, mem peak = {p7}')
        tracemalloc.reset_peak()        


        return sci_skysub_path, decorr_imgpath

def animate(frames,labels,oid,band,savename,artist='Lauren Aldoroty',savedir='/hpc/group/cosmology/lna18/roman_sim_imgs/Roman_Rubin_Sims_2024/gifs/'):
    """
    Animate a series of frames. Saves an animated GIF to the specified directory and filename. 
    """
    # NOTE: I still need to align the images to the same WCS for the animations. 
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    
    savepath = os.path.join(savedir,savename)
    metadata = dict(artist=artist)

    animate_stamps(frames,savepath,metadata,labels,staticlabel=band)

def get_templates(oid,band,n_templates=1,verbose=False):
    ra,dec,start,end = get_sn_info(oid)
    in_tab,out_tab = sn_in_or_out(oid,start,end,band)

    template_tab = out_tab[:n_templates]
    if verbose:
        print('The template images are:')
        print(template_tab)

    template_list = [dict(zip(template_tab.colnames,row)) for row in template_tab]

    return template_list

def get_science(oid,band,verbose=False):
    ra,dec,start,end = get_sn_info(oid)
    in_tab,out_tab = sn_in_or_out(oid,start,end,band)

    if verbose:
        print('The science images are:')
        print(in_tab)
        
    science_list = [dict(zip(in_tab.colnames,row)) for row in in_tab]
    
    return science_list

def skysub(infodict):
    """
    Wrapper for phrosty.imagesubtraction.sky_subtract() that allows input of
    a single dictionary for filter, pointing, and sca instead of individual
    values for each argument. Use this to enable use of multiprocessing.Pool.map.
    """
    band, pointing, sca = infodict['filter'], infodict['pointing'], infodict['sca']
    sky_subtract(band=band,pointing=pointing,sca=sca)

def run(oid,band,n_templates=1,verbose=False):

    if verbose:
        start_time = time.time()

    ra,dec,start,end = get_sn_info(oid)
    template_list = get_templates(oid,band,n_templates,verbose=verbose)
    science_list = get_science(oid,band,verbose=verbose)

    # First, unzip and sky subtract the images in their own multiprocessing pool.
    all_list = template_list + science_list
    cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
    with Pool(cpus_per_task) as pool:
        process = pool.map(skysub, all_list)
        pool.close()
        pool.join()

    if verbose:
        print('\n')
        print('********************************************************')
        print('Images have been sky-subtracted.')
        print('********************************************************')
        print('\n')

    # Now can do SFFT. 
    pairs = list(itertools.product(template_list,science_list))
    # pairslist = list((list(tup) for tup in pairs))
    # skysub_dir = '/work/lna18/imsub_out/skysub'
    # NOTE: I have fixed skysub_dir in sfft() because functools.partial()
    # was not behaving with two keyword arguments at the end. Need to sort
    # this out another time. But for now, it runs. 
    partial_sfft = partial(sfft,ra,dec,band,verbose=verbose)

    # pairs_bag = db.from_sequence(list(pairs), npartitions=cpus_per_task)
    # sfftrun = pairs_bag.map(partial_sfft,pairs_bag).compute()

    # with SharedMemoryManager() as smm:
    #     sharedlist = smm.ShareableList(pairslist)
    #     with Pool(cpus_per_task) as pool_2:
    #         process_2 = pool_2.map(partial_sfft,sharedlist)
    #         pool_2.close()
    #         pool_2.join()

    with Manager() as mgr: 
        mgr_pairs = mgr.list(pairs)
        with Pool(cpus_per_task) as pool_2:
            process_2 = pool_2.map(partial_sfft,mgr_pairs)
            pool_2.close()
            pool_2.join()

    if verbose:
        print('SFFT complete.')
        print('Run time:', time.time()-start_time)

    # NOTE: The code for animation is no longer in this because I need to 
    # rework it slightly to rotate the stamps to the same WCS anyway. 
    # For inspiration, here is the old animation code. The animations
    # should eventually be their own process pool. 
        
    # # Prepare information for animations.
    # with fits.open(ddstamppath) as ddhdu:
    #     ddimg = ddhdu[0].data
    #     dd_stamps.append(ddimg)

    # with fits.open(skysubstamppath) as skysubhdu:
    #     skysubimg = skysubhdu[0].data
    #     skysub_stamps.append(skysubimg)

    # # Make animations.
    # # They are saved with the SN ID, followed by the image used as a template.
    # animate(dd_stamps,pointings,oid,band,f'{oid}/{oid}_{band}_{t_pointing}_{t_sca}_SFFT.gif')

    # # I don't need N_templates animations for the sky subtracted-only images, so this is out of the loop.
    # animate(skysub_stamps,pointings,oid,band,f'{oid}/{oid}_{band}_RAW.gif')

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
    print("FINISHED!")

if __name__ == '__main__':
    parse_and_run()