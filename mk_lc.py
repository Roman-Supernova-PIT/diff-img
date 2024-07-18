# IMPORTS Standard:
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
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
from phrosty.utils import read_truth_txt, get_mjd, get_transient_radec, get_transient_mjd, get_mjd_info
from phrosty.photometry import ap_phot, psfmodel, psf_phot, crossmatch

###########################################################################


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
    oid, band, exptime, coord, area_eff, infodir="/hpc/group/cosmology/lna18", inputdir=".", outputdir="."
):
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

    # First, limit to images that contain the SN.
    start, end = get_transient_mjd(oid)
    in_tab = get_mjd_info(start,end,return_inverse=False)
    in_rows = np.where(np.isin(tab['pointing'],in_tab['pointing']))[0]
    in_tab = tab[in_rows]

    gs_zpt = roman.getBandpasses()[band].zeropoint
    
    ref_pointing = in_tab[0]["pointing"]
    ref_sca = in_tab[0]["sca"]

    tab = tab[1:]  # Because the first one is the reference image.
    gs_zpt = roman.getBandpasses()[band].zeropoint

    for row in in_tab:
        pointing, sca = row["pointing"], row["sca"]
        print(band, get_mjd(pointing), pointing, sca)

        # Get reference image.
        ref_path = os.path.join(inputdir, f"imsub_out/coadd/cutouts_4k/{oid}_{band}_{ref_pointing}_{ref_sca}_coadd_4kcutout.fits")
        with fits.open(ref_path) as ref_hdu:
            ref_wcs = WCS(ref_hdu[0].header)

        # Decorrelated:
        zp_imgdir = os.path.join(
            inputdir,
            f"imsub_out/science/decorr_conv_align_skysub_Roman_TDS_simple_model_{band}_{pointing}_{sca}.fits",
        )
        sndir = os.path.join(
            inputdir,
            f"imsub_out/subtract/decorr/decorr_diff_conv_align_skysub_Roman_TDS_simple_model_{band}_{pointing}_{sca}.fits",
        )
        psf_imgdir = os.path.join(
            inputdir,
            f"imsub_out/psf_final/conv_align_skysub_Roman_TDS_simple_model_{band}_{pointing}_{sca}.sfft_DCSCI.DeCorrelated.dcPSFFStack.fits",
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
            oid, band, pointing, sca, ref_wcs, exptime, area_eff, gs_zpt, nx=4088, ny=4088
        )

        # Have to clean out the rows where the value is centered on a NaN.
        # MWV: 2024-07-12  Why?  I mean, why do this based on central pixel
        # instead of waiting till you get the photometry results and on that.
        cutouts = [
            Cutout2D(zp_img, (x, y), wcs=zp_wcs, size=50, mode="partial").data
            for (x, y) in zip(stars["x"], stars["y"])
        ]
        nanrows = np.array([~np.any(np.isnan(cut)) for cut in cutouts])
        stars = stars[nanrows]

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
        star_truth_mags = (
            -2.5 * np.log10(xm["flux_truth"].data) + 2.5 * np.log10(exptime * area_eff) + gs_zpt
        )
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
        mags.append(sn_phot["mag"])
        magerr.append(sn_phot["mag_err"])
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
    savepath = os.path.join(outputdir, f"roman_sim_imgs/Roman_Rubin_Sims_2024/{oid}/{oid}_{band}_lc_coadd.csv")
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


def run(oid, band, inputdir=".", outputdir="."):
    ra, dec = get_transient_radec(oid)
    start, end = get_transient_mjd(oid)

    # oid = 20202893
    # ra = 8.037774
    # dec = -42.752337

    # oid = 20172782
    # ra = 7.551093401915147
    # dec = -44.80718106491529

    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

    exptime = {
        "F184": 901.175,
        "J129": 302.275,
        "H158": 302.275,
        "K213": 901.175,
        "R062": 161.025,
        "Y106": 302.275,
        "Z087": 101.7,
    }

    area_eff = roman.collecting_area

    lc_filename = make_lc(oid, band, exptime[band], coord, area_eff, inputdir=inputdir, outputdir=outputdir)
    plot_filename = os.path.join(outputdir, f"figs/{oid}_{band}.png")
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
        "band",
        type=str,
        choices=[None, "F184", "H158", "J129", "K213", "R062", "Y106", "Z087"],
        help="Filter to use.  None to use all available.  Overriding by --slurm_array.",
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
        help="If we're a slurm array job we're going to process the band_idx from the SLURM_ARRAY_ID.",
    )

    args = parser.parse_args()

    if args.slurm_array:
        args.band = parse_slurm()

    run(args.oid, args.band, inputdir=args.inputdir, outputdir=args.outputdir)
    print("FINISHED!")


if __name__ == "__main__":
    parse_and_run()

    oid = 20172782
