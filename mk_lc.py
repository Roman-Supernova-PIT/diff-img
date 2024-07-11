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
from astropy.table import Table, hstack
from astropy.nddata import Cutout2D
import astropy.units as u
from galsim import roman

# IMPORTS Internal:
from phrosty.utils import read_truth_txt, get_mjd, get_transient_radec, get_transient_mjd, transform_to_wcs, get_mjd_info
from phrosty.photometry import ap_phot, psfmodel, psf_phot, crossmatch

###########################################################################


###########################################################################

    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)

    exptime = {'F184': 901.175,
               'J129': 302.275,
               'H158': 302.275,
               'K213': 901.175,
               'R062': 161.025,
               'Y106': 302.275,
               'Z087': 101.7}

    area_eff = roman.collecting_area

###########################################################################

def lc(oid, band):
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
    
    tab = Table.read(f'/hpc/group/cosmology/lna18/roman_sim_imgs/Roman_Rubin_Sims_2024/{oid}/{oid}_instances_nomjd.csv')
    tab = tab[tab['filter'] == band]
    tab.sort('pointing')

    # First, limit to images that contain the SN. 
    in_tab = get_mjd_info(start,end,return_inverse=False)
    in_rows = np.where(np.isin(tab['pointing'],in_tab['pointing']))[0]
    in_tab = tab[in_rows]

    gs_zpt = roman.getBandpasses()[band].zeropoint
    
    for row in in_tab: 
        pointing, sca = row['pointing'], row['sca']
        print(band, get_mjd(pointing), pointing, sca)

        # Get reference image. 
        ref_path = f'/work/lna18/imsub_out/coadd/cutouts_4k/{oid}_{band}_{pointing}_{sca}_coadd_4kcutout.fits'
        with fits.open(ref_path) as ref_hdu:
            ref_wcs = WCS(ref_hdu[0].header)

        # Decorrelated: 
        zp_imgdir = f'/work/lna18/imsub_out/science/decorr_conv_align_skysub_Roman_TDS_simple_model_{band}_{pointing}_{sca}.fits'
        sndir = f'/work/lna18/imsub_out/subtract/decorr/decorr_diff_conv_align_skysub_Roman_TDS_simple_model_{band}_{pointing}_{sca}.fits'
        psf_imgdir = f'/work/lna18/imsub_out/psf_final/conv_align_skysub_Roman_TDS_simple_model_{band}_{pointing}_{sca}.sfft_DCSCI.DeCorrelated.dcPSFFStack.fits'

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

        # Get coordinates of stars in the aligned image. 
        orig_tab = read_truth_txt(band=band, pointing=pointing, sca=sca)
        orig_tab['x'].name, orig_tab['y'].name = 'x_orig', 'y_orig'
        orig_tab['mag'].name = 'mag_truth'
        orig_tab['mag_truth'] = -2.5*np.log10(orig_tab['flux']) + 2.5*np.log10(exptime[band]*area_eff) + gs_zpt
        transf_tab = transform_to_wcs(ref_wcs, band=band, pointing=pointing, sca=sca)
        star_idx = np.where(orig_tab['obj_type'] == 'star')[0]
        sn_idx = np.where(orig_tab['object_id'] == oid)[0]
        idx = np.append(star_idx,sn_idx)
        alltab = hstack([orig_tab, transf_tab])
        stars = alltab[idx]
        stars = stars[np.logical_and(stars['x'] < 4088, stars['x'] > 0)]
        stars = stars[np.logical_and(stars['y'] < 4088, stars['y'] > 0)]

        # Have to clean out the rows where the value is centered on a NaN.
        cutouts = [Cutout2D(zp_img, (x,y), wcs=zp_wcs, size=50, mode='partial').data for (x,y) in zip(stars['x'],stars['y'])]
        nanrows = np.array([~np.any(np.isnan(cut)) for cut in cutouts])
        stars = stars[nanrows]

        # Do PSF photometry to get the zero point. 
        # Build PSF. 
        psf = psfmodel(psf_img)
        init_params = ap_phot(zp_img,stars)
        res = psf_phot(zp_img,psf,init_params,wcs=zp_wcs)
        print('SN results from star photometry:', res[-1]['flux_fit'])

        # Crossmatch. 
        xm = crossmatch(res, stars)

        # Get the zero point. 
        star_fit_mags = -2.5*np.log10(xm['flux_fit'])
        star_truth_mags = star_truth_mags = -2.5*np.log10(xm['flux_truth'].data) + 2.5*np.log10(exptime[band]*area_eff) + gs_zpt
        zpt_mask = np.logical_and(star_truth_mags > 20, star_truth_mags < 23)
        zpt = np.nanmedian(star_truth_mags[zpt_mask] - star_fit_mags[zpt_mask]) 

        plt.figure(figsize=(6,6))
        plt.plot(star_truth_mags,star_fit_mags + zpt - star_truth_mags,
                linestyle='',marker='o')
        plt.axhline(0,color='k',linestyle='--')
        plt.xlabel('Truth')
        plt.ylabel('Fit + zpt - truth')
        plt.title(f'{band} {pointing} {sca}')
        plt.savefig(f'/hpc/group/cosmology/lna18/roman_sim_imgs/Roman_Rubin_Sims_2024/20172782/zpt_plots/zpt_{band}_{pointing}_{sca}.png', dpi=300, bbox_inches='tight')

        try: 
            # Photometry on the SN itself. 
            px_coords = sub_wcs.world_to_pixel(coord)
            forcecoords = Table([[float(px_coords[0])],[float(px_coords[1])]],names=['x','y'])
            init_sn = ap_phot(sub_img.data,forcecoords,ap_r=4)
            res_sn = psf_phot(sub_img.data,psf,init_sn,wcs=sub_wcs,forced_phot=True)

            print('SN results from SN photometry:', res_sn['flux_fit'])

            mag = -2.5*np.log10(res_sn['flux_fit'][0]) 
            mag_err = np.sqrt((1.09/res_sn['flux_fit'][0]**2*res_sn['flux_err'][0]**2))

            filters.append(band)
            pointings.append(pointing)
            scas.append(sca)
            mjds.append(get_mjd(pointing))
            mags.append(mag)
            magerr.append(mag_err)
            fluxes.append(res_sn['flux_fit'][0])
            fluxerrs.append(res_sn['flux_err'][0])
            zpts.append(zpt)
            ra_init.append(res_sn['ra_init'][0])
            dec_init.append(res_sn['dec_init'][0])
            x_init.append(res_sn['x_init'][0])
            y_init.append(res_sn['y_init'][0])
            ra_fit.append(res_sn['ra'][0])
            dec_fit.append(res_sn['dec'][0])
            x_fit.append(res_sn['x_fit'][0])
            y_fit.append(res_sn['y_fit'][0]) 

        except TypeError:
            print(f'fit_shape overlaps with edge for {band} {pointing} {sca}.')

    results = Table([filters,pointings,scas,mjds,ra_init,dec_init,x_init,y_init,ra_fit,dec_fit,x_fit,y_fit,mags,magerr,fluxes,fluxerrs,zpts],
                 names=['filter','pointing','sca','mjd','ra_init','dec_init','x_init','y_init','ra_fit','dec_fit','x_fit','y_fit','mag','magerr','flux','fluxerr','zpt'])
    savepath = f'/hpc/group/cosmology/lna18/roman_sim_imgs/Roman_Rubin_Sims_2024/{oid}/{oid}_{band}_lc_coadd.csv'
    results.write(savepath, format='csv', overwrite=True)

def main(argv):
    o_id = int(sys.argv[1])
    # b = str(sys.argv[2])
    lc(o_id,band)
    
if __name__ == '__main__':
    main(sys.argv)

# MAKE ONE LC PLOT
# Read in truth:
truthpath = f'/hpc/group/cosmology/phy-lsst/work/dms118/roman_imsim/filtered_data/filtered_data_{oid}.txt'

truth = Table.read(truthpath, names=['mjd','filter','mag'], format='ascii')[:-1]

truth['mjd'] = [row['mjd'].replace('(','') for row in truth]
truth['mag'] = [row['mag'].replace(')','') for row in truth]

truth['mjd'] = truth['mjd'].astype(float)
truth['mag'] = truth['mag'].astype(float)
truth['filter'] = [filt.upper() for filt in truth['filter']]
truth = truth[truth['filter'] == band[:1]]
truth.sort('mjd')

truthfunc = interp1d(truth['mjd'], truth['mag'], bounds_error=False)

# Get photometry.
phot_path = f'/hpc/group/cosmology/lna18/roman_sim_imgs/Roman_Rubin_Sims_2024/{oid}/{oid}_{band}_lc_coadd.csv'
phot = Table.read(phot_path)
phot.sort('mjd')

# Make residuals. 
resids = truthfunc(phot['mjd']) - (phot['mag'] + phot['zpt'])

fig, ax = plt.subplots(nrows=2,sharex=True,height_ratios=[1,0.5],figsize=(14,14))
plt.subplots_adjust(hspace=0)

ax[0].errorbar(phot['mjd'],phot['mag']+phot['zpt'],yerr=phot['magerr'],
                marker='o',linestyle='',color='mediumpurple')
ax[0].plot(truth['mjd'],truth['mag'],
                marker=None,linestyle='-',color='mediumpurple')

ax[1].axhline(0,color='k',linestyle='--')
ax[1].errorbar(phot['mjd'],resids,
                marker='o',linestyle='',color='mediumpurple')


ax[0].set_xlim(start,end)
ax[0].set_ylim(29,21)
ax[0].set_ylabel(band)
ax[1].set_xlabel('MJD')

fig.suptitle(oid,y=0.91)
plt.savefig(f'figs/{oid}_{band}_coadd.png', dpi=300, bbox_inches='tight')


# for b in bands:
#     lc(oid,b)

# lc(oid,band)

# fig = plt.figure(figsize=(14,16))
# lcidx = [0,1,2,6,7,8,12]
# resididx = [3,4,5,9,10,11,15]

# bands = ['R062','Z087','Y106','J129','H158','F184','K213']
# truthbands = ['R','Z','Y','J','H','F','K']
# colors=['mediumslateblue','darkturquoise','lightgreen','mediumvioletred','darkorange','darkgrey','black']
# cb_p = 0
# cb_t = 0

# # Read in truth:
# truthpath = f'/hpc/group/cosmology/phy-lsst/work/dms118/roman_imsim/filtered_data/filtered_data_{oid}.txt'
# truth = Table.read(truthpath, names=['mjd','filter','mag'], format='ascii')[:-1]

# truth['mjd'] = [row['mjd'].replace('(','') for row in truth]
# truth['mag'] = [row['mag'].replace(')','') for row in truth]

# truth['mjd'] = truth['mjd'].astype(float)
# truth['mag'] = truth['mag'].astype(float)
# truth['filter'] = [filt.upper() for filt in truth['filter']]

# Make plot:
# gs = GridSpec(nrows=6,ncols=3, height_ratios=[1,.33,1,.33,1,.33],width_ratios=[1,1,1],wspace=0,hspace=0)
# for i, g in enumerate(gs):
#     ax = fig.add_subplot(g)
    
#     if i in lcidx:
#         band = bands[cb_p]
#         color = colors[cb_p]
#         tband = truthbands[cb_p]
        
#         # Read in photometry:
#         path = f'/hpc/group/cosmology/lna18/roman_sim_imgs/Roman_Rubin_Sims_2024/{oid}/{oid}_{band}_lc.csv'
#         lc = Table.read(path, format='csv')
        
#         tlc = truth[truth['filter'] == tband]
#         tlc.sort('mjd')
        
#         ax.errorbar(lc['mjd'], lc['mag'] - lc['zpt'], lc['magerr'], 
#                     marker='o', linestyle='', color=color, label=f'{band}')
#         ax.plot(tlc['mjd'], tlc['mag'], color=color)
#         ax.axvline(start, color='k', linestyle='--')
#         ax.axvline(end, color='k', linestyle='--')
# #         ax.set_xlim(start,start+200)
#         ax.set_xlim(start,end)
#         ax.set_ylim(29,22)
#         ax.legend(loc='upper right',fontsize='small')
        
#         cb_p += 1
    
#     if i in resididx:
#         band = bands[cb_t]
#         color = colors[cb_t]
#         tband = truthbands[cb_t]
        
#         # Read in photometry:
#         path = f'/hpc/group/cosmology/lna18/roman_sim_imgs/Roman_Rubin_Sims_2024/{oid}/{oid}_{band}_lc.csv'
#         lc = Table.read(path, format='csv')
        
#         # Read in truth:
#         tlc = truth[truth['filter'] == tband]
#         tlc.sort('mjd')
        
#         truthfunc = interp1d(tlc['mjd'], tlc['mag'], bounds_error=False)
#         resids = lc['mag'] - lc['zpt'] - truthfunc(lc['mjd'])
        
#         ax.errorbar(lc['mjd'], resids, marker='o', linestyle='', color=color)
        
#         ax.axhline(0,color='k',linestyle='--')
#         ax.set_xlim(start,end)

#         cb_t += 1
    
# for ax in fig.get_axes():
#     ax.tick_params(bottom=False,labelbottom=False,left=False,labelleft=False)
    
# fig.get_axes()[0].tick_params(left=True,labelleft=True)
# fig.get_axes()[3].tick_params(left=True,labelleft=True)
# fig.get_axes()[6].tick_params(left=True,labelleft=True)
# fig.get_axes()[9].tick_params(left=True,labelleft=True)
# fig.get_axes()[12].tick_params(left=True,labelleft=True)
# fig.get_axes()[15].tick_params(left=True,labelleft=True)

# fig.get_axes()[15].tick_params(bottom=True,labelbottom=True)
# fig.get_axes()[16].tick_params(bottom=True,labelbottom=True)
# fig.get_axes()[17].tick_params(bottom=True,labelbottom=True)

# for i in [6,12]:
#     ax = fig.get_axes()[i]
#     plt.setp(ax.get_yticklabels()[0], visible=False)

# fig.suptitle(oid,y=0.91)

# plt.savefig(f'/hpc/group/cosmology/lna18/roman_sim_imgs/Roman_Rubin_Sims_2024/{oid}/{oid}_lc.png', dpi=300, bbox_inches='tight')


def parse_slurm():
    # SLURM
    sys.path.append(os.getcwd())

    print('taskID', taskID)

    config = {1: 'F184',
              2: 'H158',
              3: 'J129',
              4: 'K213',
              5: 'R062',
              6: 'Y106',
              7: 'Z087'}

    band = config[taskID]


def parse_and_run():
    parse = argparse.ArgumentParse(
        prog="mk_lc",
        description="Runs subtractions against a reference template for given SN"
    )
    parse.add_argument("o_id", type=int, help="ID of transient.  Used to look up hard-coded information on transient.")
    parse.add_argument("slurm_array", default=False, action="store_true", help="If we're a slurm array jobwe're going to process the band_idx from the array id.")
    
    args = parser.parse_args()

    if args.slurm:
        if os.environ['SLURM_ARRAY_TASK_ID']:
        parse.("band_idx").default = int(os.environ['SLURM_ARRAY_TASK_ID'])

    lc(args.o_id, args.band)
    print('FINISHED!')
    
if __name__ == '__main__':
    parse_and_run()
    oid = 20172782
    ra, dec = get_transient_radec(oid)
    start, end = get_transient_mjd(oid)
