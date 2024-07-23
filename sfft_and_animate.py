"""
Saves a gif and does SFFT on the specified SN. 

"""

# IMPORTS Standard:
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from copy import copy

# IMPORTS Astro:
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.visualization import ZScaleInterval
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.nddata.utils import PartialOverlapError, NoOverlapError
import astropy.units as u

# IMPORTS Internal:
from phrosty.imagesubtraction import *
from phrosty.plotting import animate_stamps
from phrosty.utils import get_transient_radec, get_transient_mjd, get_mjd_info, _build_filepath

###########################################################################

# SLURM
sys.path.append(os.getcwd())
taskID=int(os.environ['SLURM_ARRAY_TASK_ID'])

print('taskID', taskID)

# config = {1: 20172117, 
#           2: 20172199, 
#           3: 20172328, 
#           4: 20172782, 
#           5: 20173301, 
#           6: 20173305, 
#           7: 20173373, 
#           8: 20174023, 
#           9: 20174108, 
#           10: 20174118, 
#           11: 20174213, 
#           12: 20174370, 
#           13: 20174542, 
#           14: 20175077, 
#           15: 20175568, 
#           16: 20177380,
#           17: 20202893}

# oid = config[taskID]
# bands = get_roman_bands()

config = {1: 'F184',
          2: 'H158',
          3: 'J129',
          4: 'K213',
          5: 'R062',
          6: 'Y106',
          7: 'Z087'}

band = config[taskID]
print(band)
overlap_size=1000

###########################################################################

def sfft_and_animate(oid,band):
    RA, DEC = get_transient_radec(oid)
    start, end = get_transient_mjd(oid)
    coord = SkyCoord(ra=RA*u.deg,dec=DEC*u.deg)
    tab = Table.read(f'/hpc/group/cosmology/lna18/roman_sim_imgs/Roman_Rubin_Sims_2024/{oid}/{oid}_instances_nomjd.csv')
    tab.sort('pointing')
    tab = tab[tab['filter'] == band]

    # Make coadded reference image.
    # First, need all the images that don't contain the SN. 
    out_tab = get_mjd_info(start,end,return_inverse=True)
    out_rows = np.where(np.isin(tab['pointing'],out_tab['pointing']))[0]
    out_tab = tab[out_rows][:5] # Only use the first 5 images 
    filepaths = list(map(_build_filepath,[None]*len(out_tab),out_tab['filter'],out_tab['pointing'],out_tab['sca'],['image']*len(out_tab)))

    skysub_ref_init = []
    imalign_ref_init = []
    ref_idx = []
    # Sky subtract and align the reference images.
    # CAN BE PARALLELIZED.
    for i, path in enumerate(filepaths):
        skysub = sky_subtract(path=path)
        skysub_ref_init.append(skysub)
        align = imalign(template_path=skysub_ref_init[0],sci_path=skysub,force=True)
        imalign_ref_init.append(align)

        # Check that each reference tile contains an overlap_size cutout over the SN location.
        # This prevents PSF discontinuities too close to the SN. 
        with fits.open(align) as hdu:
            print('ALIGN_PATH')
            print(align)
            ref_img = hdu[0].data
            ref_wcs = WCS(hdu[0].header)
        try:
            print('COORD')
            print(coord)
            print('REF WCS')
            print(ref_wcs)
            ref_cutout = Cutout2D(ref_img,coord,overlap_size,wcs=ref_wcs,mode='strict')
            ref_idx.append(i)
        except PartialOverlapError:
            print(f'{align} does not sufficiently overlap with the SN. ')

    print('You started with', len(imalign_ref_init), 'template images for the coadd.')
    # Only keep the ones that overlap.
    skysub_ref = [skysub_ref_init[i] for i in ref_idx]
    imalign_ref = [imalign_ref_init[i] for i in ref_idx]
    filepaths_ref = [filepaths[i] for i in ref_idx]
    print('You ended with', len(imalign_ref), 'template images for the coadd after checking overlap.')

    # This is just for WCS and filename information:
    refvisit = out_tab['pointing'][0]
    refsca = out_tab['sca'][0]

    coadd_savepath = f'/work/lna18/imsub_out/coadd/{oid}_{band}_{refvisit}_{refsca}_coadd.fits'
    # if not os.path.exists(coadd_savepath):
    # Eventually don't want to rebuild them every time, but for now, maybe yeah you do. 
    # Previously aligned to filepaths[0]. 
    coadd_img_path, coadd_img_paths_all = swarp_coadd(imgpath_list=skysub_ref,
                                                          refpath=skysub_ref_init[0],
                                                          out_name=f'{oid}_{band}_{refvisit}_{refsca}_coadd.fits',
                                                          subdir='coadd')
    print('Template image successfully coadded.')

    # Save PNGs of the coadded stuff.
    with fits.open(coadd_img_path) as hdu:
        coadd_img = hdu[0].data
    vmin,vmax = ZScaleInterval().get_limits(coadd_img)
    plt.figure(figsize=(15,15))
    plt.imshow(coadd_img,vmin=vmin,vmax=vmax)
    plt.colorbar()
    plt.savefig(f'figs/{oid}_{band}_{refvisit}_{refsca}_coadd.png',dpi=300,bbox_inches='tight')
    plt.close()

    # Now, do the science images. 
    # First,  need all the images that DO contain the SN. 
    in_tab = get_mjd_info(start,end)
    in_rows = np.where(np.isin(tab['pointing'],in_tab['pointing']))[0]
    in_tab = tab[in_rows]

    # THIS CAN BE PARALLELIZED. 
    animate_panels = []
    animate_raw_panels = []
    animate_pointings = []
    n_discard = 0
    for row in in_tab:
        print('************************************************************************')
        pointing = row['pointing']
        sca = row['sca']

        print(band, pointing, sca)

        # Make a 4088x4088 cutout of the coadded reference image that perfectly overlaps with 
        # the science image. 
        ref_4k_savepath = f'/work/lna18/imsub_out/coadd/cutouts_4k/{oid}_{band}_{pointing}_{sca}_coadd_4kcutout.fits'
        ref_4k = stampmaker(RA,DEC,coadd_img_path,shape=np.array([4088,4088]),savepath=ref_4k_savepath)

        print('REF_4K_PATH')
        print(ref_4k)

        # Retrieve and rotate PSFs for PSF coaddition.
        template_psf_list = []
        for row in out_tab:
            psf = get_imsim_psf(RA,DEC,row['filter'],row['pointing'],row['sca'],size=31)
            template_psf_list.append(psf)

        coadd_psf_path, psfpaths = swarp_coadd(imgpath_list=template_psf_list,
                                                   refpath=ref_4k,
                                                   out_name=f'{oid}_{band}_{pointing}_{sca}_coadd_psf.fits',
                                                   subdir='coadd_psf')
        print('Template PSF succesfully coadded.')
        with fits.open(coadd_psf_path) as hdu:
            coadd_psf_img = hdu[0].data
            shape = coadd_psf_img.shape[0]
            cutout = Cutout2D(coadd_psf_img,(shape/2,shape/2),100).data
            vmin,vmax = ZScaleInterval().get_limits(cutout)
            plt.figure(figsize=(8,8))
            plt.imshow(coadd_psf_img,vmin=vmin,vmax=vmax)
            plt.colorbar()
            plt.savefig(f'figs/{oid}_{band}_{refvisit}_{refsca}_coadd_psf.png',dpi=300,bbox_inches='tight')
            plt.close()

        sci_skysub_path = sky_subtract(band=band,pointing=pointing,sca=sca)
        sci_imalign_path = imalign(ref_4k,sci_skysub_path,force=True)

        # Check that the image sufficiently overlaps with the SN and reference areas. 
        with fits.open(sci_imalign_path) as hdu:
            sci_img = hdu[0].data
            sci_wcs = WCS(hdu[0].header)
        try: 
            sci_cutout = Cutout2D(sci_img,coord,overlap_size,wcs=sci_wcs,mode='strict')
        except PartialOverlapError:
            print(f'{band} {pointing} {sca} does not sufficiently overlap with the SN. ')
            print('This was a SCIENCE IMAGE. It has been discarded. ')
            n_discard +=1
            continue

        # Get science image PSF and rotate it according to the coadded reference. 
        sci_imsim_psf = get_imsim_psf(RA,DEC,band,pointing,sca)
        sci_psf_path = rotate_psf(RA,DEC,sci_imsim_psf,ref_4k,force=True)

        # First convolves reference PSF on science image. 
        # Then, convolves science PSF on reference image. 
        convolvedpaths = crossconvolve(sci_imalign_path, sci_psf_path, ref_4k, coadd_psf_path, force=True)

        # Make difference image, decorrelation kernel, and decorrelated difference image. 
        print('CONVOLVEDPATHS')
        print(convolvedpaths)
        scipath, refpath = convolvedpaths
        print('SCIPATH AFTER CROSS CONVOLVE')
        print(scipath)
        print('REFPATH AFTER CROSS CONVOLVE')
        print(refpath)
        diff, soln = difference(scipath, refpath, sci_psf_path, coadd_psf_path, backend='Numpy', force=True)
        print('DIFFERENCE IMAGE PATH')
        print(diff)        
        # NOTE: Decorrelation kernel only uses the science and reference images to calculate
        # sky background, so don't use the cross-convolved images. Use the sky-subtracted ones.
        # (And in the case of the science image, the aligned and sky-subtracted one.)
        dcker_path = decorr_kernel(sci_imalign_path, ref_4k, sci_psf_path, coadd_psf_path, diff, soln)
        print('DECORR KERNEL PATH')
        print(dcker_path)
        dcimg_path = decorr_img(diff, dcker_path)
        print('DECORR IMAGE PATH')
        print(dcimg_path)

        # Save difference image stamps. 
        decorr_savepath = f'/work/lna18/imsub_out/subtract/decorr/finalstamps/1K/finalstamp_{RA}_{DEC}_Roman_TDS_simple_model_{band}_{pointing}_{sca}.fits'
        decorr_path = stampmaker(RA,DEC,dcimg_path,savepath=decorr_savepath,shape=np.array([1000,1000]))
        finalhdu = fits.open(decorr_path)
        finalstamp = finalhdu[0].data

        decorr_small_savepath = f'/work/lna18/imsub_out/subtract/decorr/finalstamps/50/finalstamp_{RA}_{DEC}_Roman_TDS_simple_model_{band}_{pointing}_{sca}.fits'
        decorr_small_path = stampmaker(RA,DEC,dcimg_path,savepath=decorr_small_savepath,shape=np.array([50,50]))
        finalhdu_small = fits.open(decorr_small_path)
        finalstamp_small = finalhdu_small[0].data

        # Also make stamps of sky subtracted original science image, while we're at it.
        rawstamp_savepath = f'/work/lna18/imsub_out/rawstamps/1K/{oid}_{band}_{pointing}_{sca}.fits'
        rawstamp_path = stampmaker(RA,DEC,sci_imalign_path,savepath=rawstamp_savepath,shape=np.array([1000,1000]))
        rawhdu = fits.open(rawstamp_path)
        rawstamp = rawhdu[0].data

        rawstamp_small_savepath = f'/work/lna18/imsub_out/rawstamps/50/{oid}_{band}_{pointing}_{sca}.fits'
        rawstamp_small_path = stampmaker(RA,DEC,sci_imalign_path,savepath=rawstamp_small_savepath,shape=np.array([50,50]))
        rawhdu_small = fits.open(rawstamp_small_path)
        rawstamp_small = rawhdu_small[0].data

        # Detour: Need to take the cross convolved science image and apply the decorrelation
        # kernel. We get the zpt from this image. 
        decorr_scipath = decorr_img(scipath,dcker_path,imgtype='science')

        # Calculate PSF
        psfpath = calc_psf(scipath,refpath,sci_psf_path,coadd_psf_path,dcker_path)

        animate_panels.append(finalstamp_small)
        animate_raw_panels.append(rawstamp_small)
        animate_pointings.append(pointing)

        zscale = ZScaleInterval()
        z1, z2 = zscale.get_limits(rawstamp)
        fig, ax = plt.subplots(figsize=(5,5))
        im = ax.imshow(rawstamp, cmap='Greys', vmin=z1, vmax=z2)
        ax.text(0.05,0.95,pointing,color='white',transform=ax.transAxes,va='top',ha='left')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        plt.savefig(f'figs/sfft/RAW_{oid}_{band}_{pointing}_{sca}.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(n_discard, 'science images were discarded due to insufficent overlap.')
    # Make animation. 
    savepath = f'/hpc/group/cosmology/lna18/roman_sim_imgs/Roman_Rubin_Sims_2024/gifs/{oid}/{oid}_{band}_SFFT.gif'
    if not os.path.exists(f'/hpc/group/cosmology/lna18/roman_sim_imgs/Roman_Rubin_Sims_2024/gifs/{oid}/'):
        os.mkdir(f'/hpc/group/cosmology/lna18/roman_sim_imgs/Roman_Rubin_Sims_2024/gifs/{oid}/')
    
    metadata = dict(artist='Lauren Aldoroty')
    animate_stamps(animate_panels, savepath, metadata, labels=animate_pointings, staticlabel=band)

    savepath = f'/hpc/group/cosmology/lna18/roman_sim_imgs/Roman_Rubin_Sims_2024/gifs/{oid}/{oid}_{band}_RAW.gif'
    metadata = dict(artist='Lauren Aldoroty')
    animate_stamps(animate_raw_panels, savepath, metadata, labels=animate_pointings, staticlabel=band)
    
def main(argv):
    o_id = int(sys.argv[1])
    # b = str(sys.argv[2])
    sfft_and_animate(o_id,band)
    
if __name__ == '__main__':
    main(sys.argv)