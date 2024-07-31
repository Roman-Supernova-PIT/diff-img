# IMPORTS Standard:
import argparse
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# IMPORTS Astro:
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.visualization import ZScaleInterval
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS, NoConvergence
from astropy.wcs.utils import skycoord_to_pixel
from astropy.nddata.utils import PartialOverlapError, NoOverlapError
import astropy.units as u

# IMPORTS Internal:
from phrosty.imagesubtraction import sky_subtract, imalign, get_imsim_psf, rotate_psf, crossconvolve, difference, decorr_kernel, decorr_img, stampmaker
from phrosty.plotting import animate_stamps
from phrosty.utils import get_transient_radec, get_transient_mjd, get_mjd_info, _build_filepath, get_fitsobj

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

def check_overlap(ra,dec,imgpath,data_ext=0,overlap_size=1000,verbose=False,show_cutout=False):
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

def sfft(ra,dec,band,sci_pointing,sci_sca,
                     t_pointing,t_sca,verbose=False):
    """
    Run SFFT steps. This includes:
    1. Aligning template image to science image
    2. Preprocessing the science image
    3. Cross-convolution
    4. Differencing
    5. Generating decorrelation kernel
    6. Applying decorrelation kernel to difference image and cross-convolved science image

    """

    sci_skysub_path = sky_subtract(band=band,pointing=sci_pointing,sca=sci_sca)
    t_filepath = _build_filepath(None,band,t_pointing,t_sca,'image')
    t_skysub = sky_subtract(path=t_filepath)
    t_align = imalign(template_path=sci_skysub_path,sci_path=t_skysub) # NOTE: This is correct, not flipped.

    template_overlap = check_overlap(ra,dec,t_align,verbose=verbose)
    science_overlap = check_overlap(ra,dec,sci_skysub_path,verbose=verbose)

    if not template_overlap or not science_overlap:
        if verbose:
            print(f'Images {band} {sci_pointing} {sci_sca} and {band} {t_pointing} {t_sca} do not sufficiently overlap to do image subtraction.')

        return None, None

    else:
        sci_psf_path = get_imsim_psf(ra,dec,band,sci_pointing,sci_sca)
        if verbose:
            print('\n')
            print('Path to science PSF:')
            print(sci_psf_path)
            print('\n')

        t_imsim_psf = get_imsim_psf(ra,dec,band,t_pointing,t_sca)
        t_psf_path = rotate_psf(ra,dec,t_imsim_psf,sci_skysub_path,force=True)
        if verbose:
            print('\n')
            print('Path to template PSF:')
            print(t_psf_path)
            print('\n')

        sci_conv, temp_conv = crossconvolve(sci_skysub_path,sci_psf_path,t_align,t_imsim_psf,force=True)
        if verbose:
            print('\n')
            print('Path to cross-convolved science image:')
            print(sci_conv)
            print('Path to cross-convolved template image:')
            print(temp_conv)
            print('\n')

        diff_savename = f'{band}_{sci_pointing}_{sci_sca}_-_{t_pointing}_{t_sca}.fits' # 'diff_' gets appended to the beginning of this
        diff, soln = difference(sci_conv,temp_conv,sci_psf_path,t_psf_path,savename=diff_savename,backend='Numpy',force=True)
        if verbose:
            print('\n')
            print('Path to differenced image:')
            print(diff)
            print('\n')
        
        dcker_savename = f'dcker_{band}_{sci_pointing}_{sci_sca}_-_{t_pointing}_{t_sca}.fits'
        dcker_path = decorr_kernel(sci_skysub_path,t_skysub,sci_psf_path,t_psf_path,diff,soln,savename=dcker_savename)
        if verbose:
            print('\n')
            print('Path to decorrelation kernel:')
            print(dcker_path)
            print('\n')
        
        decorr_imgpath = decorr_img(diff,dcker_path)
        if verbose:
            print('\n')
            print('Path to final decorrelated differenced image:')
            print(decorr_imgpath)
            print('\n')

        zpt_savename = f'zptimg_{band}_{sci_pointing}_{sci_sca}_-_{t_pointing}_{t_sca}.fits'
        zpt_imgpath = decorr_img(sci_conv,dcker_path,savename=zpt_savename)
        if verbose:
            print('\n')
            print('Path to zeropoint image:')
            print(zpt_imgpath)
            print('\n')

        decorr_psf_savename = f'psf_{band}_{sci_pointing}_{sci_sca}_-_{t_pointing}_{t_sca}.fits'
        decorr_psfpath = decorr_img(sci_psf_path,dcker_path,savename=decorr_psf_savename)
        if verbose:
            print('\n')
            print('Path to decorrelated PSF (use for photometry):')
            print(decorr_psfpath)
            print('\n')

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

def run(oid,band,n_templates=1,verbose=False):

    ra,dec,start,end = get_sn_info(oid)
    in_tab,out_tab = sn_in_or_out(oid,start,end,band)

    template_tab = out_tab[:n_templates]
    if verbose:
        print('The template images are:')
        print(template_tab)

    for t_row in template_tab:
        t_pointing, t_sca = t_row['pointing'], t_row['sca']

        dd_stamps = []
        skysub_stamps = []
        pointings = []
        for row in in_tab:
            print('*************************************************************')
            sci_pointing, sci_sca = row['pointing'], row['sca']
            print(band,sci_pointing,sci_sca,'-',band,t_pointing,t_sca)

            skysubimgpath, ddimgpath = sfft(ra,dec,band,sci_pointing,sci_sca,t_pointing,t_sca,verbose=verbose)
            if skysubimgpath is not None:
                # Make stamps.
                dd_stamp_savepath = f'/work/lna18/imsub_out/decorr/stamps/dd_stamp_{oid}_{band}_{sci_pointing}_{sci_sca}_-_{t_pointing}_{t_sca}.fits'
                skysub_stamp_savepath = f'/work/lna18/imsub_out/skysub/stamps/skysub_stamp_{oid}_{band}_{sci_pointing}_{sci_sca}_-_{t_pointing}_{t_sca}.fits'

                ddstamppath = stampmaker(ra,dec,ddimgpath,savepath=dd_stamp_savepath,shape=np.array([100,100]))
                skysubstamppath = stampmaker(ra,dec,ddimgpath,savepath=skysub_stamp_savepath,shape=np.array([100,100]))
            
                # Prepare information for animations.
                pointings.append(sci_pointing)
                with fits.open(ddstamppath) as ddhdu:
                    ddimg = ddhdu[0].data
                    dd_stamps.append(ddimg)

                with fits.open(skysubstamppath) as skysubhdu:
                    skysubimg = skysubhdu[0].data
                    skysub_stamps.append(skysubimg)

            # Make animations.
            # They are saved with the SN ID, followed by the image used as a template.
            animate(dd_stamps,pointings,oid,band,f'{oid}/{oid}_{band}_{t_pointing}_{t_sca}_SFFT.gif')

    # I don't need N_templates animations for the sky subtracted-only images, so this is out of the loop.
    animate(skysub_stamps,pointings,oid,band,f'{oid}/{oid}_{band}_RAW.gif')

def parse_slurm():
    """
    Turn a SLURM array ID into a band. 
    """
    sys.path.append(os.getcwd())
    taskID = int(os.environ["SLURM_ARRAY_TASK_ID"])

    print("taskID", taskID)

    config = {1: "F184", 2: "H158", 3: "J129", 4: "K213", 5: "R062", 6: "Y106", 7: "Z087"}

    band = config[taskID]
    print('band', band)

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
        'n_templates',
        type=int,
        default=1,
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