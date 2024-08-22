# IMPORTS Standard:
import argparse
import os
import sys
import itertools
from collections import defaultdict
from functools import partial
from multiprocessing import Pool, Manager
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d

# Make numpy stop thread hogging. 
# I don't think I need this if I have it in the batch file, right? 
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import numpy as np

# IMPORTS Astro:
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.table import Table, hstack, unique
from astropy.nddata import Cutout2D
import astropy.units as u
from galsim import roman

# IMPORTS Internal:
from phrosty.utils import read_truth_txt, get_mjd, get_transient_radec, get_transient_mjd, get_mjd_info, get_exptime, get_transient_info, transient_in_or_out, get_science, get_templates, get_exptime
from phrosty.photometry import ap_phot, psfmodel, psf_phot, crossmatch

###########################################################################

simsdir = os.getenv('SIMS_DIR', None)
assert simsdir is not None, 'You need to set SIMSDIR as an environment variable.'

infodir = os.getenv('SN_INFO_DIR', None)
assert infodir is not None, 'You need to set SN_INFO_DIR as an environment variable.'

dia_out_dir = os.getenv('DIA_OUT_DIR', None)
assert dia_out_dir is not None, 'You need to set DIA_INFO_DIR as an environment variable.'

lc_out_dir = os.getenv('LC_OUT_DIR', None)
assert lc_out_dir is not None, 'You need to set LC_OUT_DIR as an environment variable.'

snid_lc_dir = os.getenv('SNID_LC_DIR', None)
assert snid_lc_dir is not None, 'You need to set SNID_LC_DIR as an environment variable.'

###########################################################################

def get_galsim_values(band):
    exptime = get_exptime(band)
    area_eff = roman.collecting_area
    gs_zpt = roman.getBandpasses()[band].zeropoint

    return {'exptime': exptime, 'area_eff': area_eff, 'gs_zpt': gs_zpt}

def get_stars(band,truthpath,nx=4088,ny=4088,transform=False,wcs=None):
    """
    Get the stars in the science images.

    Optional to transform to another WCS. 
    """
    truth_tab = read_truth_txt(path=truthpath)
    truth_tab['mag'].name = 'mag_truth'
    truth_tab['flux'].name = 'flux_truth'

    if transform:
        assert wcs is not None, 'You need to provide a WCS to transform to!'
        truth_tab['x'].name, truth_tab['y'].name = 'x_orig', 'y_orig'
        worldcoords = SkyCoord(ra=truth_tab['ra'] * u.deg, dec=truth_tab['dec'] * u.deg)
        x, y = skycoord_to_pixel(worldcoords, wcs)
        truth_tab['x'] = x
        truth_tab['y'] = y

    idx = np.where(truth_tab['obj_type'] == 'star')[0]
    stars = truth_tab[idx]
    stars = stars[np.logical_and(stars["x"] < nx, stars["x"] > 0)]
    stars = stars[np.logical_and(stars["y"] < ny, stars["y"] > 0)]

    return stars

def get_psf(psfpath):
    """
    Get the PSF in the form of a FittableImageModel.
    """

    with fits.open(psfpath) as hdu:
        psf = psfmodel(hdu[0].data)

    return psf

def get_zpt(zptimg,psf,band,stars):
    """
    Get the zeropoint based on the stars in the convolved, decorrelated science image. 
    """
    
    # First, need to do photometry on the stars.
    init_params = ap_phot(zptimg, stars)
    final_params = psf_phot(zptimg, psf, init_params)

    # Do not need to cross match. Can just merge tables because they
    # will be in the same order. 
    photres = hstack([stars, final_params])

    # Get the zero point. 
    galsim_vals = get_galsim_values(band)
    star_fit_mags = -2.5 * np.log10(photres['flux_fit'])
    star_truth_mags = -2.5 * np.log10(photres['flux_truth']) + 2.5 * np.log10(galsim_vals['exptime'] * galsim_vals['area_eff']) + galsim_vals['gs_zpt']

    # Eventually, this should be a S/N cut, not a mag cut. 
    zpt_mask = np.logical_and(star_truth_mags > 20, star_truth_mags < 23)
    zpt = np.nanmedian(star_truth_mags[zpt_mask] - star_fit_mags[zpt_mask])

    return zpt

def phot_at_coords(img, psf, pxcoords=(50, 50), ap_r=4):
    """
    Do photometry at forced set of pixel coordinates. 
    """

    forcecoords = Table([[float(pxcoords[0])], [float(pxcoords[1])]], names=["x", "y"])
    init = ap_phot(img,forcecoords,ap_r=ap_r)
    final = psf_phot(img,psf,init,forced_phot=True)

    flux = final['flux_fit'][0]
    flux_err = final['flux_err'][0]
    mag = -2.5 * np.log10(final["flux_fit"][0])
    mag_err = (2.5 / np.log(10)) * np.abs(final["flux_err"][0] / final["flux_fit"][0])

    results_dict = {
                    'flux_fit': flux,
                    'flux_fit_err': flux_err,
                    'mag_fit': mag,
                    'mag_fit_err': mag_err
                    }

    return results_dict

def make_phot_info_dict(oid, band, pair_info, pxcoords=(50, 50), ap_r=4):

    ra, dec, start, end = get_transient_info(oid)
    template_info, sci_info = pair_info
    sci_pointing, sci_sca = sci_info['pointing'], sci_info['sca']
    template_pointing, template_sca = template_info['pointing'], template_info['sca']

    # Get MJD. 
    mjd = get_mjd(sci_pointing)

    # Load in the difference image stamp.
    diff_img_stamp_path = os.path.join(dia_out_dir, f'stamps/stamp_{ra}_{dec}_diff_Roman_TDS_simple_model_{band}_{sci_pointing}_{sci_sca}_-_{band}_{template_pointing}_{template_sca}.fits')
    with fits.open(diff_img_stamp_path) as diff_hdu:
        diffimg = diff_hdu[0].data

    # Load in the decorrelated PSF.
    psfpath = os.path.join(dia_out_dir, f'decorr/psf_{band}_{sci_pointing}_{sci_sca}_-_{band}_{template_pointing}_{template_sca}.fits')
    psf = get_psf(psfpath)
    results_dict = phot_at_coords(diffimg, psf, pxcoords=pxcoords, ap_r=ap_r)

    # Get the zero point from the decorrelated, convolved science image.
    # First, get the table of known stars.
    truthpath = os.path.join(simsdir, f'RomanTDS/truth/{band}/{sci_pointing}/Roman_TDS_index_{band}_{sci_pointing}_{sci_sca}.txt')
    stars = get_stars(band, truthpath)

    # Now, calculate the zero point based on those stars. 
    zptimg_path = os.path.join(dia_out_dir, f'decorr/zptimg_{band}_{sci_pointing}_{sci_sca}_-_{band}_{template_pointing}_{template_sca}.fits')
    with fits.open(zptimg_path) as hdu:
        zptimg = hdu[0].data
    zpt = get_zpt(zptimg, psf, band, stars)

    # Add additional info to the results dictionary so it can be merged into a nice file later.
    results_dict['ra'] = ra
    results_dict['dec'] = dec
    results_dict['mjd'] = mjd
    results_dict['filter'] = band
    results_dict['pointing'] = sci_pointing
    results_dict['sca'] = sci_sca
    results_dict['template_pointing'] = template_pointing
    results_dict['template_sca'] = template_sca
    results_dict['zpt'] = zpt

    return results_dict

# def get_truth_lcs(oid, band, mjd):


def make_lc_plot(oid, band, start, end, phot_path=None):
    # MAKE ONE LC PLOT
    # Read in truth:
    # truthpath = f"/hpc/group/cosmology/phy-lsst/work/dms118/roman_imsim/filtered_data/filtered_data_{oid}.txt"

    # truth = Table.read(truthpath, names=["mjd", "filter", "mag"], format="ascii")[:-1]

    # truth["mjd"] = [row["mjd"].replace("(", "") for row in truth]
    # truth["mag"] = [row["mag"].replace(")", "") for row in truth]

    # truth["mjd"] = truth["mjd"].astype(float)
    # truth["mag"] = truth["mag"].astype(float)
    # truth["filter"] = [filt.upper() for filt in truth["filter"]]
    # truth = truth[truth["filter"] == band[:1]]
    # truth.sort("mjd")

    # truthfunc = interp1d(truth["mjd"], truth["mag"], bounds_error=False)

    # Get photometry.
    phot = Table.read(phot_path)
    phot.sort("mjd")

    template_pointing = str(np.unique(phot['template_pointing'].data)[0])
    template_sca = str(np.unique(phot['template_sca'].data)[0])

    print('TEMPLATE POINTING, TEMPLATE SCA')
    print(template_pointing, template_sca)

    # Make residuals.
    # resids = truthfunc(phot["mjd"]) - (phot["mag"] + phot["zpt"])

    fig, ax = plt.subplots(nrows=2, sharex=True, height_ratios=[1, 0.5], figsize=(14, 14))
    plt.subplots_adjust(hspace=0)

    ax[0].errorbar(
        phot["mjd"],
        phot["mag_fit"] + phot["zpt"],
        yerr=phot["mag_fit_err"],
        marker="o",
        linestyle="",
        color="mediumpurple",
    )
    # ax[0].plot(truth["mjd"], truth["mag"], marker=None, linestyle="-", color="mediumpurple")

    ax[1].axhline(0, color="k", linestyle="--")
    # ax[1].errorbar(phot["mjd"], resids, marker="o", linestyle="", color="mediumpurple")

    ax[0].set_xlim(start, end)
    ax[0].set_ylim(29, 21)
    ax[0].set_ylabel(band)
    ax[1].set_xlabel("MJD")

    fig.suptitle(oid, y=0.91)

    savedir = os.path.join(lc_out_dir,f'figs/{oid}')
    os.makedirs(savedir, exist_ok=True)
    savename = f'{oid}_{band}_{template_pointing}_{template_sca}.png'
    savepath = os.path.join(savedir,savename)
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close()


def split_table(oid, band, tab, template_info):
    template_pointing, template_sca = template_info
    tab = tab[tab['template_pointing'] == template_pointing]
    savedir = os.path.join(lc_out_dir,f'data/{str(oid)}')
    os.makedirs(savedir, exist_ok=True)
    savename = f'{oid}_{band}_{template_pointing}_{template_sca}.csv'
    savepath = os.path.join(savedir,savename)
    tab.write(savepath, format='csv', overwrite=True)

    return savepath

def run(oid,band,n_templates=1,verbose=False):
    template_list = get_templates(oid,band,infodir,n_templates,verbose=verbose)
    science_list = get_science(oid,band,infodir,verbose=verbose)
    pairs = list(itertools.product(template_list,science_list))

    partial_make_phot_info_dict = partial(make_phot_info_dict,oid,band,n_templates=n_templates)

    cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
    with Manager() as mgr:
        mgr_pairs = mgr.list(pairs)
        with Pool(cpus_per_task) as pool:
            res = pool.map(partial_make_phot_info_dict, mgr_pairs)
            pool.close()
            pool.join()

    results_dict = defaultdict(list)
    for d in res:
        for key, value in d.items():
            results_dict[key].append(value)

    results_tab = Table(results_dict)
    results_tab.sort('pointing')
    results_savedir = os.path.join(lc_out_dir,f'data/{str(oid)}')
    os.makedirs(results_savedir, exist_ok=True)
    results_savename = f'{oid}_{band}_all.csv'
    results_savepath = os.path.join(results_savedir,results_savename)
    results_tab.write(results_savepath, format='csv', overwrite=True)

    partial_split_table = partial(split_table,oid,band,results_tab)

    pointings = unique(results_tab['template_pointing','template_sca'], keys='template_pointing').as_array()
    print('POINTINGS')
    print(pointings)
    with Pool(n_templates) as pool_2:
        res_2 = pool_2.map(partial_split_table,pointings)
        pool_2.close()
        pool_2.join()

    ra,dec,start,end = get_transient_info(oid)
    for path in res_2:
        make_lc_plot(oid, band, start, end, path)

    

# def calc_sn_photometry(img, wcs, psf, coord):
#     """
#     Do photometry on the SN itself
#     """
#     # Photometry on the SN itself.
#     px_coords = wcs.world_to_pixel(coord)
#     forcecoords = Table([[float(px_coords[0])], [float(px_coords[1])]], names=["x", "y"])
#     init_sn = ap_phot(img.data, forcecoords, ap_r=4)
#     sn_phot = psf_phot(img.data, psf, init_sn, wcs=wcs, forced_phot=True)

#     print("SN results from SN photometry:", sn_phot["flux_fit"])

#     mag = -2.5 * np.log10(sn_phot["flux_fit"][0])
#     mag_err = (2.5 / np.log(10)) * np.abs(sn_phot["flux_err"][0] / sn_phot["flux_fit"][0])
#     sn_phot["mag"] = mag
#     sn_phot["mag_err"] = mag_err

# def make_lc(
#     oid, band, exptime, area_eff, in_tab, t_pointing, t_sca, inputdir=dia_out_dir, outputdir=lc_out_dir
# ):
#     """
#     Make one light curve for one series of single-epoch subtractions 
#     (i.e., one SN using DIA all with the same template).
#     """

#     ra,dec,start,end = get_transient_info(oid)
#     coord = SkyCoord(ra=ra*u.deg,dec=dec*u.deg)

#     filters = []
#     pointings = []
#     scas = []
#     mjds = []
#     mags = []
#     magerr = []
#     fluxes = []
#     fluxerrs = []
#     zpts = []
#     ra_init = []
#     dec_init = []
#     x_init = []
#     y_init = []
#     ra_fit = []
#     dec_fit = []
#     x_fit = []
#     y_fit = []

#     transient_info_file = os.path.join(
#         infodir, f"roman_sim_imgs/Roman_Rubin_Sims_2024/{oid}/{oid}_instances_nomjd.csv"
#     )
#     tab = Table.read(transient_info_file)
#     tab = tab[tab["filter"] == band]
#     tab.sort("pointing")

#     gs_zpt = roman.getBandpasses()[band].zeropoint

#     tab = tab[1:]  # Because the first one is the reference image.
#     gs_zpt = roman.getBandpasses()[band].zeropoint

#     for row in in_tab:
#         pointing, sca = row["pointing"], row["sca"]
#         print(band, get_mjd(pointing), pointing, sca)

#         # Open necessary images.
#         # Zeropoint image:
#         zp_imgdir = os.path.join(
#             inputdir,
#             f"decorr/zptimg_{band}_{pointing}_{sca}_-_{t_pointing}_{t_sca}.fits",
#         )

#         # SN difference image:
#         sndir = os.path.join(
#             inputdir,
#             f"decorr/decorr_diff_{band}_{pointing}_{sca}_-_{t_pointing}_{t_sca}.fits",
#         )

#         # PSF: 
#         psf_imgdir = os.path.join(
#             inputdir,
#             f"decorr/psf_{band}_{pointing}_{sca}_-_{t_pointing}_{t_sca}.fits",
#         )

#         input_paths = [zp_imgdir,sndir,psf_imgdir]
#         paths_exist = [os.path.exists(path) for path in input_paths]

#         if all(paths_exist):
#             # Sky-subtracted field for generating the ZP.
#             zp_hdu = fits.open(zp_imgdir)
#             zp_img = zp_hdu[0].data
#             zp_wcs = WCS(zp_hdu[0].header)

#             # Subtracted stamp (SN only)
#             sub_hdu = fits.open(sndir)
#             sub_img = sub_hdu[0].data
#             sub_wcs = WCS(sub_hdu[0].header)

#             # PSF
#             psf_hdu = fits.open(psf_imgdir)
#             psf_img = psf_hdu[0].data

#             stars = get_star_truth_coordinates_in_aligned_images(
#                 oid, band, pointing, sca, zp_wcs, exptime, area_eff, gs_zpt, nx=4088, ny=4088
#             )

#             # Have to clean out the rows where the value is centered on a NaN.
#             # MWV: 2024-07-12  Why?  I mean, why do this based on central pixel
#             # instead of waiting till you get the photometry results and on that.
#             cutouts = [
#                 Cutout2D(zp_img, (x, y), wcs=zp_wcs, size=50, mode="partial").data
#                 for (x, y) in zip(stars["x"], stars["y"])
#             ]
#             nanrows = np.array([~np.any(np.isnan(cut)) for cut in cutouts])
#             stars = stars[nanrows]

#             # Do PSF photometry to get the zero point.
#             # Build PSF.
#             psf = psfmodel(psf_img)
#             init_params = ap_phot(zp_img, stars)
#             res = psf_phot(zp_img, psf, init_params, wcs=zp_wcs)
#             # MWV: 2024-07-12:  What is this printing?
#             # The instrumental flux for the SN?
#             print("SN results from star photometry:", res[-1]["flux_fit"])

#             # Crossmatch.
#             xm = crossmatch(res, stars)

#             # Get the zero point.
#             star_fit_mags = -2.5 * np.log10(xm["flux_fit"])
#             star_truth_mags = -2.5 * np.log10(xm["flux_truth"].data) + 2.5 * np.log10(exptime * area_eff) + gs_zpt
#             zpt_mask = np.logical_and(star_truth_mags > 20, star_truth_mags < 23)
#             zpt = np.nanmedian(star_truth_mags[zpt_mask] - star_fit_mags[zpt_mask])

#             try:
#                 print('SUBTRACTED IMAGE, WCS, PSF, COORD')
#                 print(sub_img)
#                 print(sub_wcs)
#                 print(coord)
#                 sn_phot = calc_sn_photometry(sub_img, sub_wcs, psf, coord)
#             # MWV: 2024-07-12: Why is this the error we expect?
#             except TypeError as te:
#                 print(te)
#                 print(f"fit_shape overlaps with edge for {band} {pointing} {sca}.")
#                 continue
#             except ValueError as ve:
#                 print(ve)
#                 print(f'There are probably NaNs where an object should be in {band} {pointing} {sca}.')
#                 continue


#             filters.append(band)
#             pointings.append(pointing)
#             scas.append(sca)
#             mjds.append(get_mjd(pointing))
#             mags.append(sn_phot["mag"][0])
#             magerr.append(sn_phot["mag_err"][0])
#             fluxes.append(sn_phot["flux_fit"][0])
#             fluxerrs.append(sn_phot["flux_err"][0])
#             zpts.append(zpt)
#             ra_init.append(sn_phot["ra_init"][0])
#             dec_init.append(sn_phot["dec_init"][0])
#             x_init.append(sn_phot["x_init"][0])
#             y_init.append(sn_phot["y_init"][0])
#             ra_fit.append(sn_phot["ra"][0])
#             dec_fit.append(sn_phot["dec"][0])
#             x_fit.append(sn_phot["x_fit"][0])
#             y_fit.append(sn_phot["y_fit"][0])
#         else:
#             continue

#     results = Table(
#         [
#             filters,
#             pointings,
#             scas,
#             mjds,
#             ra_init,
#             dec_init,
#             x_init,
#             y_init,
#             ra_fit,
#             dec_fit,
#             x_fit,
#             y_fit,
#             mags,
#             magerr,
#             fluxes,
#             fluxerrs,
#             zpts,
#         ],
#         names=[
#             "filter",
#             "pointing",
#             "sca",
#             "mjd",
#             "ra_init",
#             "dec_init",
#             "x_init",
#             "y_init",
#             "ra_fit",
#             "dec_fit",
#             "x_fit",
#             "y_fit",
#             "mag",
#             "magerr",
#             "flux",
#             "fluxerr",
#             "zpt",
#         ],
#     )
#     savepath = os.path.join(
#         outputdir, f"data/{oid}/{oid}_{band}_{t_pointing}_{t_sca}_lc.csv"
#     )
#     os.makedirs(os.path.dirname(savepath), exist_ok=True)
#     results.write(savepath, format="csv", overwrite=True)

#     return savepath

# def get_star_truth_coordinates_in_aligned_images(
#     oid, band, pointing, sca, 
#     ref_wcs, exptime, area_eff, zpt, nx=4088, ny=4088
# ):
#     """
#     Get coordinates of stars in the aligned image.

#     Uses the truth table from the simulation to get RA, Dec.
#     Then the WCS from the image to transform to x, y.
#     """
#     orig_tab = read_truth_txt(band=band, pointing=pointing, sca=sca)
#     orig_tab["x"].name, orig_tab["y"].name = "x_orig", "y_orig"
#     orig_tab["mag"].name = "mag_truth"
#     orig_tab["mag_truth"] = -2.5 * np.log10(orig_tab["flux"]) + 2.5 * np.log10(exptime * area_eff) + zpt
#     worldcoords = SkyCoord(ra=orig_tab["ra"] * u.deg, dec=orig_tab["dec"] * u.deg)
#     x, y = skycoord_to_pixel(worldcoords, ref_wcs)

#     alltab = orig_tab.copy()
#     alltab["x"] = x
#     alltab["y"] = y

#     star_idx = np.where(orig_tab["obj_type"] == "star")[0]
#     sn_idx = np.where(orig_tab["object_id"] == oid)[0]
#     idx = np.append(star_idx, sn_idx)

#     stars = alltab[idx]
#     stars = stars[np.logical_and(stars["x"] < 4088, stars["x"] > 0)]
#     stars = stars[np.logical_and(stars["y"] < 4088, stars["y"] > 0)]

#     return stars


# def plot_star_truth_vs_fit_mag(
#     star_truth_mags, star_fit_mags, zpt, 
#     band, pointing, sca, 
#     plot_filename=os.path.join(lc_out_dir,"figs/zpt.png"), 
#     figsize=(6, 6)
# ):
#     """
#     Plot the star truth mag versus the measured mag and calculated zeropoint.
#     """
#     plt.figure(figsize=figsize)
#     plt.plot(star_truth_mags, star_fit_mags + zpt - star_truth_mags, linestyle="", marker="o")
#     plt.axhline(0, color="k", linestyle="--")
#     plt.xlabel("Truth")
#     plt.ylabel("Fit + zpt - truth")
#     plt.title(f"{band} {pointing} {sca}")
#     plt.savefig(
#         plot_filename,
#         dpi=300,
#         bbox_inches="tight",
#     )
#     plt.close()


# def plot_lc_with_all_bands(oid,start,end):
#     """
#     Plot a LC with all bands for a SN.
#     """
#     fig = plt.figure(figsize=(14,16))
#     lcidx = [0,1,2,6,7,8,12]
#     resididx = [3,4,5,9,10,11,15]

#     bands = ['R062','Z087','Y106','J129','H158','F184','K213']
#     truthbands = ['R','Z','Y','J','H','F','K']
#     colors=['mediumslateblue','darkturquoise','lightgreen','mediumvioletred','darkorange','darkgrey','black']
#     cb_p = 0
#     cb_t = 0

#     # Read in truth:
#     truthpath = f'/hpc/group/cosmology/phy-lsst/work/dms118/roman_imsim/filtered_data/filtered_data_{oid}.txt'
#     truth = Table.read(truthpath, names=['mjd','filter','mag'], format='ascii')[:-1]

#     truth['mjd'] = [row['mjd'].replace('(','') for row in truth]
#     truth['mag'] = [row['mag'].replace(')','') for row in truth]

#     truth['mjd'] = truth['mjd'].astype(float)
#     truth['mag'] = truth['mag'].astype(float)
#     truth['filter'] = [filt.upper() for filt in truth['filter']]

#     gs = GridSpec(nrows=6,ncols=3, height_ratios=[1,.33,1,.33,1,.33],width_ratios=[1,1,1],wspace=0,hspace=0)

#     for i, g in enumerate(gs):
#         ax = fig.add_subplot(g)

#         if i in lcidx:
#             band = bands[cb_p]
#             color = colors[cb_p]
#             tband = truthbands[cb_p]

#             # Read in photometry:
#             photdir = os.path.join(lc_out_dir,f'data/{oid}')
#             photdata_filename = f'{oid}_{band}_lc.csv'
#             path = os.path.join(photdir,photdata_filename)
#             lc = Table.read(path, format='csv')

#             tlc = truth[truth['filter'] == tband]
#             tlc.sort('mjd')

#             ax.errorbar(lc['mjd'], lc['mag'] - lc['zpt'], lc['magerr'],
#                         marker='o', linestyle='', color=color, label=f'{band}')
#             ax.plot(tlc['mjd'], tlc['mag'], color=color)
#             ax.axvline(start, color='k', linestyle='--')
#             ax.axvline(end, color='k', linestyle='--')
#     #         ax.set_xlim(start,start+200)
#             ax.set_xlim(start,end)
#             ax.set_ylim(29,22)
#             ax.legend(loc='upper right',fontsize='small')

#             cb_p += 1

#         if i in resididx:
#             band = bands[cb_t]
#             color = colors[cb_t]
#             tband = truthbands[cb_t]

#             # Read in photometry:
#             photdir = os.path.join(lc_out_dir,f'data/{oid}')
#             photdata_filename = f'{oid}_{band}_lc.csv'
#             path = os.path.join(photdir,photdata_filename)
#             lc = Table.read(path, format='csv')

#             # Read in truth:
#             tlc = truth[truth['filter'] == tband]
#             tlc.sort('mjd')

#             truthfunc = interp1d(tlc['mjd'], tlc['mag'], bounds_error=False)
#             resids = lc['mag'] - lc['zpt'] - truthfunc(lc['mjd'])

#             ax.errorbar(lc['mjd'], resids, marker='o', linestyle='', color=color)

#             ax.axhline(0,color='k',linestyle='--')
#             ax.set_xlim(start,end)

#             cb_t += 1

#     for ax in fig.get_axes():
#         ax.tick_params(bottom=False,labelbottom=False,left=False,labelleft=False)

#     fig.get_axes()[0].tick_params(left=True,labelleft=True)
#     fig.get_axes()[3].tick_params(left=True,labelleft=True)
#     fig.get_axes()[6].tick_params(left=True,labelleft=True)
#     fig.get_axes()[9].tick_params(left=True,labelleft=True)
#     fig.get_axes()[12].tick_params(left=True,labelleft=True)
#     fig.get_axes()[15].tick_params(left=True,labelleft=True)

#     fig.get_axes()[15].tick_params(bottom=True,labelbottom=True)
#     fig.get_axes()[16].tick_params(bottom=True,labelbottom=True)
#     fig.get_axes()[17].tick_params(bottom=True,labelbottom=True)

#     for i in [6,12]:
#         ax = fig.get_axes()[i]
#         plt.setp(ax.get_yticklabels()[0], visible=False)

#     fig.suptitle(oid,y=0.91)

#     savefig_dir = os.path.join(lc_out_dir,f'figs/{oid}')
#     savefig_name = f'{oid}_lc.png'
#     savefig_path = os.path.join(savefig_dir,savefig_name)
#     plt.savefig(savefig_path, dpi=300, bbox_inches='tight')


#     return sn_phot




# def run(oid, band, n_templates=1, inputdir=dia_out_dir, outputdir=lc_out_dir):

#     template_list = get_templates(oid,band,infodir,n_templates,verbose=verbose)
#     science_list = get_science(oid,band,infodir,verbose=verbose)
#     pairs = list(itertools.product(template_list,science_list))

#     exptime = get_exptime(band)
#     area_eff = roman.collecting_area

#     for row in template_tab:
#         t_pointing, t_sca = row['pointing'], row['sca']
#         lc_filename = make_lc(oid, band, exptime, area_eff, in_tab, t_pointing,t_sca, inputdir=inputdir, outputdir=outputdir)
#         plot_filename = os.path.join(outputdir, f"figs/{oid}/{oid}_{band}_{t_pointing}_{t_sca}.png")
#         os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
#         make_lc_plot(oid, band, start, end, lc_filename)


def parse_slurm():
    # SLURM
    sys.path.append(os.getcwd())
    taskID = int(os.environ["SLURM_ARRAY_TASK_ID"])

    print("taskID", taskID)

    config = {1: "F184", 2: "H158", 3: "J129", 4: "K213", 5: "R062", 6: "Y106", 7: "Z087"}

    band = config[taskID]
    return band


def parse_and_run():
    parser = argparse.ArgumentParser(
        prog="mk_lc", description="Photometry for a given SN"
    )
    parser.add_argument(
        "oid", type=int, help="ID of transient.  Used to look up information on transient."
    )
    parser.add_argument(
        "--band",
        type=str,
        default=None,
        choices=[None, "F184", "H158", "J129", "K213", "R062", "Y106", "Z087"],
        help="Filter to use.  None to use all available.  Overridden by --slurm_array.",
    )

    parser.add_argument(
        '--n-templates',
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
    print("Finished making LCs!")


if __name__ == "__main__":
    parse_and_run()
