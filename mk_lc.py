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
from astropy.table import Table, join, unique, vstack
from astropy.nddata import Cutout2D
import astropy.units as u

from photutils.psf import IntegratedGaussianPRF

from galsim import roman

# IMPORTS Internal:
from phrosty.utils import read_truth_txt, get_roman_bands, get_mjd, get_transient_radec, get_transient_mjd, get_mjd_info, get_exptime, get_transient_info, transient_in_or_out, get_science, get_templates, get_exptime
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

def get_stars(truthpath,nx=4088,ny=4088,transform=False,wcs=None):
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

    if not transform:
        truth_tab['x'] -= 1
        truth_tab['y'] -= 1

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

def get_zpt(zptimg,psf,band,stars,ap_r=4,ap_phot_only=False,zpt_plot=True,oid=None,sci_pointing=None,sci_sca=None):
    """
    Get the zeropoint based on the stars.
    """

    # First, need to do photometry on the stars.
    init_params = ap_phot(zptimg, stars, ap_r=ap_r)
    final_params = psf_phot(zptimg, psf, init_params, forced_phot=True)

    # Do not need to cross match. Can just merge tables because they
    # will be in the same order.
    photres = join(stars,init_params,keys=['object_id','ra','dec','realized_flux','flux_truth','mag_truth','obj_type'])
    if not ap_phot_only:
        photres = join(photres,final_params,keys=['id'])

    # Get the zero point.
    galsim_vals = get_galsim_values(band)
    star_fit_mags = -2.5 * np.log10(photres['flux_fit'])
    star_truth_mags = - 2.5 * np.log10(photres['flux_truth']) + galsim_vals['gs_zpt'] + 2.5 * np.log10(galsim_vals['exptime'] * galsim_vals['area_eff'])

    # Eventually, this should be a S/N cut, not a mag cut.
    zpt_mask = np.logical_and(star_truth_mags > 20, star_truth_mags < 23)
    zpt = np.nanmedian(star_truth_mags[zpt_mask] - star_fit_mags[zpt_mask])

    if zpt_plot:
        assert oid is not None, 'If zpt_plot=True, oid must be provided.'
        assert sci_pointing is not None, 'If zpt_plot=True, sci_pointing must be provided.'
        assert sci_sca is not None, 'If zpt_plot=True, sci_sca must be provided.'

        savedir = os.path.join(lc_out_dir,f'figs/{oid}/zpt_plots')
        os.makedirs(savedir, exist_ok=True)
        savepath = os.path.join(savedir,f'zpt_stars_{band}_{sci_pointing}_{sci_sca}.png')

        plt.figure(figsize=(8,8))
        yaxis = star_fit_mags + zpt - star_truth_mags

        plt.plot(star_truth_mags,yaxis,marker='o',linestyle='')
        plt.axhline(0,linestyle='--',color='k')
        plt.xlabel('Truth mag')
        plt.ylabel('Fit mag - zpt + truth mag')
        plt.title(f'{band} {sci_pointing} {sci_sca}')
        plt.savefig(savepath,dpi=300,bbox_inches='tight')
        plt.close()

        savepath = os.path.join(savedir,f'hist_truth-fit_{band}_{sci_pointing}_{sci_sca}.png')
        plt.hist(star_truth_mags[zpt_mask] - star_fit_mags[zpt_mask])
        plt.title(f'{band} {sci_pointing} {sci_sca}')
        plt.xlabel('star_truth_mags[zpt_mask] - star_fit_mags[zpt_mask]')
        plt.savefig(savepath,dpi=300,bbox_inches='tight')
        plt.close()

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

    print('RESULTS_DICT flux_fit')
    print('flux_fit')
    print(flux)

    return results_dict

def make_phot_info_dict(oid, band, pair_info, ap_r=4):

    ra, dec, start, end = get_transient_info(oid)
    template_info, sci_info = pair_info
    sci_pointing, sci_sca = sci_info['pointing'], sci_info['sca']
    template_pointing, template_sca = template_info['pointing'], template_info['sca']
    mjd = get_mjd(sci_pointing)

    # Make sure there's a difference image stamp to do photometry on.
    # TODO: Bring stamp making back, and change this path back to actually run on a stamp (less memory)
    # diff_img_stamp_path = os.path.join(dia_out_dir, f'stamps/stamp_{ra}_{dec}_decorr_diff_Roman_TDS_simple_model_{band}_{sci_pointing}_{sci_sca}_-_{band}_{template_pointing}_{template_sca}.fits')
    diff_img_stamp_path = os.path.join(dia_out_dir,f'decorr/decorr_diff_{band}_{sci_pointing}_{sci_sca}_-_{band}_{template_pointing}_{template_sca}.fits')
    if os.path.exists(diff_img_stamp_path):
        # Load in the difference image stamp.
        with fits.open(diff_img_stamp_path) as diff_hdu:
            diffimg = diff_hdu[0].data
            wcs = WCS(diff_hdu[0].header)

        # Load in the decorrelated PSF.
        psfpath = os.path.join(dia_out_dir, f'decorr/psf_{band}_{sci_pointing}_{sci_sca}_-_{band}_{template_pointing}_{template_sca}.fits')
        psf = get_psf(psfpath)
        coord = SkyCoord(ra=ra*u.deg,dec=dec*u.deg)
        pxcoords = skycoord_to_pixel(coord,wcs)
        results_dict = phot_at_coords(diffimg, psf, pxcoords=pxcoords, ap_r=ap_r)

        # Get the zero point from the decorrelated, convolved science image.
        # First, get the table of known stars.

        truthpath = os.path.join(simsdir, f'RomanTDS/truth/{band}/{sci_pointing}/Roman_TDS_index_{band}_{sci_pointing}_{sci_sca}.txt')
        stars = get_stars(truthpath)

        # Now, calculate the zero point based on those stars.
        zptimg_path = os.path.join(dia_out_dir, f'decorr/zptimg_{band}_{sci_pointing}_{sci_sca}_-_{band}_{template_pointing}_{template_sca}.fits')
        with fits.open(zptimg_path) as hdu:
            zptimg = hdu[0].data
        zpt = get_zpt(zptimg, psf, band, stars, oid=oid, sci_pointing=sci_pointing, sci_sca=sci_sca)

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

    else:
        print(f'Post-processed image files for {band}_{sci_pointing}_{sci_sca}_-_{band}_{template_pointing}_{template_sca} do not exist. Skipping.')
        results_dict = {}
        results_dict['ra'] = ra
        results_dict['dec'] = dec
        results_dict['mjd'] = mjd
        results_dict['filter'] = band
        results_dict['pointing'] = sci_pointing
        results_dict['sca'] = sci_sca
        results_dict['template_pointing'] = template_pointing
        results_dict['template_sca'] = template_sca
        results_dict['zpt'] = np.nan
        results_dict['flux_fit'] = np.nan
        results_dict['flux_fit_err'] = np.nan
        results_dict['mag_fit'] = np.nan
        results_dict['mag_fit_err'] = np.nan

    return results_dict

def get_truth_lc(oid, band):

    shortened_bands = {filt: str(filt[0]) for filt in get_roman_bands()}

    base_path = os.path.join(snid_lc_dir,'ROMAN+LSST_NONIaMODEL0-')
    for i in range(1,31):
        suffix = f"{i:04d}"  # format to get 0001, 0002, etc.
        path_head = base_path+suffix+"_HEAD.FITS.gz"
        path_phot = base_path+suffix+"_PHOT.FITS.gz"

        with fits.open(path_head) as hdu:
            data_head = hdu[1].data
            snid_array = data_head['SNID']
            ptr_obs_min = data_head['PTROBS_MIN']
            ptr_obs_max = data_head['PTROBS_MAX']

            try:
                idx = list(snid_array).index(str(oid))
                start = ptr_obs_min[idx]
                end = ptr_obs_max[idx]+1

                with fits.open(path_phot) as hdu_phot:
                    data_phot = Table(hdu_phot[1].data)
                    extracted_data = data_phot[start:end]
                    bands_col = extracted_data['BAND'].data  # There is a trailing whitespace in each entry in this column!
                    sb = shortened_bands[band]
                    bands_idx = [j for j, b in enumerate(bands_col) if b[0] == sb]  # [0] for the trailing whitespace!
                    band_filtered_data = extracted_data[bands_idx]

                return band_filtered_data

            except ValueError:
                continue

def make_lc_plot(oid, band, start, end, phot_path=None):

    # First, get truth LC.
    truthdata = get_truth_lc(oid,band)
    truthfunc = interp1d(truthdata["MJD"], truthdata["SIM_MAGOBS"], bounds_error=False)

    # Get photometry.
    phot = Table.read(phot_path)
    phot.sort("mjd")

    template_pointing = str(np.unique(phot['template_pointing'].data)[0])
    template_sca = str(np.unique(phot['template_sca'].data)[0])

    # Make residuals.
    resids = truthfunc(phot["mjd"]) - (phot["mag_fit"] + phot["zpt"])

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
    ax[0].plot(truthdata["MJD"], truthdata["SIM_MAGOBS"], marker=None, linestyle="-", color="mediumpurple")

    ax[1].axhline(0, color="k", linestyle="--")
    ax[1].errorbar(phot["mjd"], resids, marker="o", linestyle="", color="mediumpurple")

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

def run(oid, band, sci_list_path, template_list_path, cpus_per_task=1, verbose=False):
    science_tab = Table.read(sci_list_path)
    science_tab = science_tab[science_tab['filter'] == band]
    science_list = [dict(zip(science_tab.colnames,row)) for row in science_tab]

    template_tab = Table.read(template_list_path)
    template_tab = template_tab[template_tab['filter'] == band]
    template_list = [dict(zip(template_tab.colnames,row)) for row in template_tab]

    pairs = list(itertools.product(template_list,science_list))

    partial_make_phot_info_dict = partial(make_phot_info_dict,oid,band)

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
    with Pool(cpus_per_task) as pool_2:
        res_2 = pool_2.map(partial_split_table,pointings)
        pool_2.close()
        pool_2.join()

    ra,dec,start,end = get_transient_info(oid)
    for path in res_2:
        make_lc_plot(oid, band, start, end, path)


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
    print("Finished making LCs!")


if __name__ == "__main__":
    parse_and_run()
