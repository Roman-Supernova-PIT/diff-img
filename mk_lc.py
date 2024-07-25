# IMPORTS Standard:
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d

# IMPORTS Astro:
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.table import Table
from astropy.nddata import Cutout2D
import astropy.units as u
from galsim import roman

# IMPORTS Internal:
from phrosty.utils import read_truth_txt, get_mjd, get_transient_radec, get_transient_mjd, get_mjd_info, get_exptime
from phrosty.photometry import ap_phot, psfmodel, psf_phot, crossmatch

###########################################################################

def set_sn(oid):
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
    """
    transient_info_file = os.path.join(
        infodir, f'roman_sim_imgs/Roman_Rubin_Sims_2024/{oid}/{oid}_instances_nomjd.csv'
    )

    tab = Table.read(transient_info_file)
    tab.sort('pointing')
    tab = tab[tab['filter'] == band]

    in_tab_all = get_mjd_info(start,end)
    in_rows = np.where(np.isin(tab['pointing'],in_tab_all['pointing']))[0]
    in_tab = tab[in_rows]

    out_tab_all = get_mjd_info(start,end,return_inverse=True)
    out_rows = np.where(np.isin(tab['pointing'],out_tab_all['pointing']))[0]
    out_tab = tab[out_rows]

    return in_tab, out_tab

def get_star_truth_coordinates_in_aligned_images(
    oid, band, pointing, sca, ref_wcs, exptime, area_eff, zpt, nx=4088, ny=4088
):
    """
    Get coordinates of stars in the aligned image.

    Uses the truth table from the simulation to get RA, Dec.
    Then the WCS from the image to transform to x, y.
    """
    orig_tab = read_truth_txt(band=band, pointing=pointing, sca=sca)
    orig_tab["x"].name, orig_tab["y"].name = "x_orig", "y_orig"
    orig_tab["mag"].name = "mag_truth"
    orig_tab["mag_truth"] = -2.5 * np.log10(orig_tab["flux"]) + 2.5 * np.log10(exptime * area_eff) + zpt
    worldcoords = SkyCoord(ra=orig_tab["ra"] * u.deg, dec=orig_tab["dec"] * u.deg)
    x, y = skycoord_to_pixel(worldcoords, ref_wcs)

    alltab = orig_tab.copy()
    alltab["x"] = x
    alltab["y"] = y

    star_idx = np.where(orig_tab["obj_type"] == "star")[0]
    sn_idx = np.where(orig_tab["object_id"] == oid)[0]
    idx = np.append(star_idx, sn_idx)

    stars = alltab[idx]
    stars = stars[np.logical_and(stars["x"] < 4088, stars["x"] > 0)]
    stars = stars[np.logical_and(stars["y"] < 4088, stars["y"] > 0)]

    return stars


def plot_star_truth_vs_fit_mag(
    star_truth_mags, star_fit_mags, zpt, band, pointing, sca, plot_filename="zpt.png", figsize=(6, 6)
):
    """
    Plot the star truth mag versus the measured mag and calculated zeropoint.
    """
    plt.figure(figsize=figsize)
    plt.plot(star_truth_mags, star_fit_mags + zpt - star_truth_mags, linestyle="", marker="o")
    plt.axhline(0, color="k", linestyle="--")
    plt.xlabel("Truth")
    plt.ylabel("Fit + zpt - truth")
    plt.title(f"{band} {pointing} {sca}")
    plt.savefig(
        plot_filename,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_lc_with_all_bands(oid,start,end):
    """
    Plot a LC with all bands for a SN.
    """
    fig = plt.figure(figsize=(14,16))
    lcidx = [0,1,2,6,7,8,12]
    resididx = [3,4,5,9,10,11,15]

    bands = ['R062','Z087','Y106','J129','H158','F184','K213']
    truthbands = ['R','Z','Y','J','H','F','K']
    colors=['mediumslateblue','darkturquoise','lightgreen','mediumvioletred','darkorange','darkgrey','black']
    cb_p = 0
    cb_t = 0

    # Read in truth:
    truthpath = f'/hpc/group/cosmology/phy-lsst/work/dms118/roman_imsim/filtered_data/filtered_data_{oid}.txt'
    truth = Table.read(truthpath, names=['mjd','filter','mag'], format='ascii')[:-1]

    truth['mjd'] = [row['mjd'].replace('(','') for row in truth]
    truth['mag'] = [row['mag'].replace(')','') for row in truth]

    truth['mjd'] = truth['mjd'].astype(float)
    truth['mag'] = truth['mag'].astype(float)
    truth['filter'] = [filt.upper() for filt in truth['filter']]

    gs = GridSpec(nrows=6,ncols=3, height_ratios=[1,.33,1,.33,1,.33],width_ratios=[1,1,1],wspace=0,hspace=0)

    for i, g in enumerate(gs):
        ax = fig.add_subplot(g)

        if i in lcidx:
            band = bands[cb_p]
            color = colors[cb_p]
            tband = truthbands[cb_p]

            # Read in photometry:
            path = f'/hpc/group/cosmology/lna18/roman_sim_imgs/Roman_Rubin_Sims_2024/{oid}/{oid}_{band}_lc.csv'
            lc = Table.read(path, format='csv')

            tlc = truth[truth['filter'] == tband]
            tlc.sort('mjd')

            ax.errorbar(lc['mjd'], lc['mag'] - lc['zpt'], lc['magerr'],
                        marker='o', linestyle='', color=color, label=f'{band}')
            ax.plot(tlc['mjd'], tlc['mag'], color=color)
            ax.axvline(start, color='k', linestyle='--')
            ax.axvline(end, color='k', linestyle='--')
    #         ax.set_xlim(start,start+200)
            ax.set_xlim(start,end)
            ax.set_ylim(29,22)
            ax.legend(loc='upper right',fontsize='small')

            cb_p += 1

        if i in resididx:
            band = bands[cb_t]
            color = colors[cb_t]
            tband = truthbands[cb_t]

            # Read in photometry:
            path = f'/hpc/group/cosmology/lna18/roman_sim_imgs/Roman_Rubin_Sims_2024/{oid}/{oid}_{band}_lc.csv'
            lc = Table.read(path, format='csv')

            # Read in truth:
            tlc = truth[truth['filter'] == tband]
            tlc.sort('mjd')

            truthfunc = interp1d(tlc['mjd'], tlc['mag'], bounds_error=False)
            resids = lc['mag'] - lc['zpt'] - truthfunc(lc['mjd'])

            ax.errorbar(lc['mjd'], resids, marker='o', linestyle='', color=color)

            ax.axhline(0,color='k',linestyle='--')
            ax.set_xlim(start,end)

            cb_t += 1

    for ax in fig.get_axes():
        ax.tick_params(bottom=False,labelbottom=False,left=False,labelleft=False)

    fig.get_axes()[0].tick_params(left=True,labelleft=True)
    fig.get_axes()[3].tick_params(left=True,labelleft=True)
    fig.get_axes()[6].tick_params(left=True,labelleft=True)
    fig.get_axes()[9].tick_params(left=True,labelleft=True)
    fig.get_axes()[12].tick_params(left=True,labelleft=True)
    fig.get_axes()[15].tick_params(left=True,labelleft=True)

    fig.get_axes()[15].tick_params(bottom=True,labelbottom=True)
    fig.get_axes()[16].tick_params(bottom=True,labelbottom=True)
    fig.get_axes()[17].tick_params(bottom=True,labelbottom=True)

    for i in [6,12]:
        ax = fig.get_axes()[i]
        plt.setp(ax.get_yticklabels()[0], visible=False)

    fig.suptitle(oid,y=0.91)

    plt.savefig(f'/hpc/group/cosmology/lna18/roman_sim_imgs/Roman_Rubin_Sims_2024/{oid}/{oid}_lc.png', dpi=300, bbox_inches='tight')


def calc_sn_photometry(img, wcs, psf, coord):
    """
    Do photometry on the SN itself
    """
    # Photometry on the SN itself.
    px_coords = wcs.world_to_pixel(coord)
    forcecoords = Table([[float(px_coords[0])], [float(px_coords[1])]], names=["x", "y"])
    init_sn = ap_phot(img.data, forcecoords, ap_r=4)
    sn_phot = psf_phot(img.data, psf, init_sn, wcs=wcs, forced_phot=True)

    print("SN results from SN photometry:", sn_phot["flux_fit"])

    mag = -2.5 * np.log10(sn_phot["flux_fit"][0])
    mag_err = (2.5 / np.log(10)) * np.abs(sn_phot["flux_err"][0] / sn_phot["flux_fit"][0])
    sn_phot["mag"] = mag
    sn_phot["mag_err"] = mag_err

    return sn_phot


def make_lc(
    oid, band, exptime, coord, area_eff, in_tab, t_pointing, t_sca, infodir="/hpc/group/cosmology/lna18", inputdir=".", outputdir="."
):
    """
    Make one light curve for one series of single-epoch subtractions (i.e., one SN using DIA all with the same template).
    """

    if not os.path.exists(outputdir):
        os.path.mkdir(outputdir)

    filters = []
    pointings = []
    scas = []
    mjds = []
    mags = []
    magerr = []
    fluxes = []
    fluxerrs = []
    zpts = []
    ra_init = []
    dec_init = []
    x_init = []
    y_init = []
    ra_fit = []
    dec_fit = []
    x_fit = []
    y_fit = []

    transient_info_file = os.path.join(
        infodir, f"roman_sim_imgs/Roman_Rubin_Sims_2024/{oid}/{oid}_instances_nomjd.csv"
    )
    tab = Table.read(transient_info_file)
    tab = tab[tab["filter"] == band]
    tab.sort("pointing")

    gs_zpt = roman.getBandpasses()[band].zeropoint

    tab = tab[1:]  # Because the first one is the reference image.
    gs_zpt = roman.getBandpasses()[band].zeropoint

    for row in in_tab:
        pointing, sca = row["pointing"], row["sca"]
        print(band, get_mjd(pointing), pointing, sca)

        # Open necesary images.
        # Zeropoint image:
        zp_imgdir = os.path.join(
            inputdir,
            f"imsub_out/decorr/decorr_zptimg_{band}_{pointing}_{sca}_-_{t_pointing}_{t_sca}.fits",
        )
        # SN difference image:
        sndir = os.path.join(
            inputdir,
            f"imsub_out/decorr/decorr_diff_{band}_{pointing}_{sca}_-_{t_pointing}_{t_sca}.fits",
        )
        # PSF: 
        psf_imgdir = os.path.join(
            inputdir,
            f"imsub_out/decorr/decorr_psf_{band}_{pointing}_{sca}_-_{t_pointing}_{t_sca}.fits",
        )

        # Sky-subtracted field for generating the ZP.
        zp_hdu = fits.open(zp_imgdir)
        zp_img = zp_hdu[0].data
        zp_wcs = WCS(zp_hdu[0].header)

        # Subtracted stamp (SN only)
        sub_hdu = fits.open(sndir)
        sub_img = sub_hdu[0].data
        sub_wcs = WCS(sub_hdu[0].header)

        # PSF
        psf_hdu = fits.open(psf_imgdir)
        psf_img = psf_hdu[0].data

        stars = get_star_truth_coordinates_in_aligned_images(
            oid, band, pointing, sca, zp_wcs, exptime, area_eff, gs_zpt, nx=4088, ny=4088
        )

        # Have to clean out the rows where the value is centered on a NaN.
        # MWV: 2024-07-12  Why?  I mean, why do this based on central pixel
        # instead of waiting till you get the photometry results and on that.
        # cutouts = [
        #     Cutout2D(zp_img, (x, y), wcs=zp_wcs, size=50, mode="partial").data
        #     for (x, y) in zip(stars["x"], stars["y"])
        # ]
        # nanrows = np.array([~np.any(np.isnan(cut)) for cut in cutouts])
        # stars = stars[nanrows]

        # Do PSF photometry to get the zero point.
        # Build PSF.
        psf = psfmodel(psf_img)
        init_params = ap_phot(zp_img, stars)
        res = psf_phot(zp_img, psf, init_params, wcs=zp_wcs)
        # MWV: 2024-07-12:  What is this printing?
        # The instrumental flux for the SN?
        print("SN results from star photometry:", res[-1]["flux_fit"])

        # Crossmatch.
        xm = crossmatch(res, stars)

        # Get the zero point.
        star_fit_mags = -2.5 * np.log10(xm["flux_fit"])
        star_truth_mags = -2.5 * np.log10(xm["flux_truth"].data) + 2.5 * np.log10(exptime * area_eff) + gs_zpt
        zpt_mask = np.logical_and(star_truth_mags > 20, star_truth_mags < 23)
        zpt = np.nanmedian(star_truth_mags[zpt_mask] - star_fit_mags[zpt_mask])

        plot_filename = os.path.join(
            outputdir,
            f"roman_sim_imgs/Roman_Rubin_Sims_2024/{oid}/zpt_plots/zpt_{band}_{pointing}_{sca}.png",
        )
        # Make sure directory of plot_filename file exists.
        os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
        plot_star_truth_vs_fit_mag(
            star_truth_mags, star_fit_mags, zpt, band, pointing, sca, plot_filename=plot_filename
        )

        try:
            sn_phot = calc_sn_photometry(sub_img, sub_wcs, psf, coord)
        # MWV: 2024-07-12: Why is this the error we expect?
        except TypeError:
            print(f"fit_shape overlaps with edge for {band} {pointing} {sca}.")
            continue

        filters.append(band)
        pointings.append(pointing)
        scas.append(sca)
        mjds.append(get_mjd(pointing))
        mags.append(sn_phot["mag"][0])
        magerr.append(sn_phot["mag_err"][0])
        fluxes.append(sn_phot["flux_fit"][0])
        fluxerrs.append(sn_phot["flux_err"][0])
        zpts.append(zpt)
        ra_init.append(sn_phot["ra_init"][0])
        dec_init.append(sn_phot["dec_init"][0])
        x_init.append(sn_phot["x_init"][0])
        y_init.append(sn_phot["y_init"][0])
        ra_fit.append(sn_phot["ra"][0])
        dec_fit.append(sn_phot["dec"][0])
        x_fit.append(sn_phot["x_fit"][0])
        y_fit.append(sn_phot["y_fit"][0])

    results = Table(
        [
            filters,
            pointings,
            scas,
            mjds,
            ra_init,
            dec_init,
            x_init,
            y_init,
            ra_fit,
            dec_fit,
            x_fit,
            y_fit,
            mags,
            magerr,
            fluxes,
            fluxerrs,
            zpts,
        ],
        names=[
            "filter",
            "pointing",
            "sca",
            "mjd",
            "ra_init",
            "dec_init",
            "x_init",
            "y_init",
            "ra_fit",
            "dec_fit",
            "x_fit",
            "y_fit",
            "mag",
            "magerr",
            "flux",
            "fluxerr",
            "zpt",
        ],
    )
    savepath = os.path.join(
        outputdir, f"roman_sim_imgs/Roman_Rubin_Sims_2024/{oid}/{oid}_{band}_{t_pointing}_{t_sca}_lc.csv"
    )
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    results.write(savepath, format="csv", overwrite=True)

    return savepath


def make_lc_plot(oid, band, start, end, lc_filename, plot_filename):
    # MAKE ONE LC PLOT
    # Read in truth:
    truthpath = f"/hpc/group/cosmology/phy-lsst/work/dms118/roman_imsim/filtered_data/filtered_data_{oid}.txt"

    truth = Table.read(truthpath, names=["mjd", "filter", "mag"], format="ascii")[:-1]

    truth["mjd"] = [row["mjd"].replace("(", "") for row in truth]
    truth["mag"] = [row["mag"].replace(")", "") for row in truth]

    truth["mjd"] = truth["mjd"].astype(float)
    truth["mag"] = truth["mag"].astype(float)
    truth["filter"] = [filt.upper() for filt in truth["filter"]]
    truth = truth[truth["filter"] == band[:1]]
    truth.sort("mjd")

    truthfunc = interp1d(truth["mjd"], truth["mag"], bounds_error=False)

    # Get photometry.
    phot_path = os.path.join(lc_filename)
    phot = Table.read(phot_path)
    phot.sort("mjd")

    # Make residuals.
    resids = truthfunc(phot["mjd"]) - (phot["mag"] + phot["zpt"])

    fig, ax = plt.subplots(nrows=2, sharex=True, height_ratios=[1, 0.5], figsize=(14, 14))
    plt.subplots_adjust(hspace=0)

    ax[0].errorbar(
        phot["mjd"],
        phot["mag"] + phot["zpt"],
        yerr=phot["magerr"],
        marker="o",
        linestyle="",
        color="mediumpurple",
    )
    ax[0].plot(truth["mjd"], truth["mag"], marker=None, linestyle="-", color="mediumpurple")

    ax[1].axhline(0, color="k", linestyle="--")
    ax[1].errorbar(phot["mjd"], resids, marker="o", linestyle="", color="mediumpurple")

    ax[0].set_xlim(start, end)
    ax[0].set_ylim(29, 21)
    ax[0].set_ylabel(band)
    ax[1].set_xlabel("MJD")

    fig.suptitle(oid, y=0.91)
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()


def run(oid, band, n_templates=1, inputdir=".", outputdir="."):

    ra,dec,start,end = set_sn(oid)
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    in_tab,out_tab = sn_in_or_out(oid,start,end,band)
    template_tab = out_tab[:n_templates]

    exptime = get_exptime(band)
    area_eff = roman.collecting_area

    for row in template_tab:
        t_pointing, t_sca = row['pointing'], row['sca']
        lc_filename = make_lc(oid, band, exptime, coord, area_eff, in_tab, t_pointing,t_sca, inputdir=inputdir, outputdir=outputdir)
        plot_filename = os.path.join(outputdir, f"figs/{oid}_{band}_{t_pointing}_{t_sca}.png")
        os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
        make_lc_plot(oid, band, start, end, lc_filename, plot_filename)


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
        prog="mk_lc", description="Runs subtractions against a reference template for given SN"
    )
    parser.add_argument(
        "oid", type=int, help="ID of transient.  Used to look up hard-coded information on transient."
    )
    parser.add_argument(
        "--band",
        type=str,
        default=None,
        choices=[None, "F184", "H158", "J129", "K213", "R062", "Y106", "Z087"],
        help="Filter to use.  None to use all available.  Overriding by --slurm_array.",
    )

    parser.add_argument(
        'n_templates',
        type=int,
        default=1,
        help='Number of template images to use.'
    )

    parser.add_argument(
        "--inputdir",
        type=str,
        default=".",
        help="Location of input information from sfft_and_animate.py",
    )
    parser.add_argument(
        "--outputdir",
        type=str,
        default=".",
        help="Location of output information from sfft_and_animate.py",
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

    run(args.oid, args.band, inputdir=args.inputdir, outputdir=args.outputdir)
    print("FINISHED!")


if __name__ == "__main__":
    parse_and_run()

    oid = 20172782
