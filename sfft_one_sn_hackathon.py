import nvtx

import os
import cupy

from astropy.io import fits

from phrosty.utils import set_logger

import preprocess
import sfftdiff
import postprocess
import mk_lc

sn = '20172782'
band = 'R062'
sci_list_path = "/phrosty_etal/20172782_instances_two_science.csv"
templ_list_path = "/phrosty_etal/20172782_one_templates.csv"

ncpus = 3

logger = set_logger( "wrapper", "wrapper" )

with nvtx.annotate( "get info and images", color="yellow" ):
    objinfo, template_imgs, science_imgs = preprocess.get_transient_info_and_images( sn,
                                                                                     band,
                                                                                     sci_list_path,
                                                                                     templ_list_path )
with nvtx.annotate( "sky sub", color="red" ):
    template_imgs, science_imgs = preprocess.sky_sub_all_images( template_imgs, science_imgs, cpus_per_task=ncpus )
with nvtx.annotate( "psfs", color="green" ):
    template_imgs, science_imgs = preprocess.get_psfs( objinfo, template_imgs, science_imgs, cpus_per_task=ncpus )

for templ_info in template_imgs.values():
    for sci_info in science_imgs.values():
        
        with nvtx.annotate( "align_and_preconvolve", color="purple" ):
            sfftifier = sfftdiff.align_and_pre_convolve( objinfo['ra'], objinfo['dec'], band,
                                                         ( templ_info, sci_info ) )
        # Save and free

        with nvtx.annotate( "subtraction", color=0x00ffff ):
            sfftifier.sfft_subtraction()
        # Save and free

        with nvtx.annotate( "find_decorrelation", color=0x8800ff ):
            sfftifier.find_decorrelation()
        # Save and free

        with nvtx.annotate( "decorrelate", color=0x00ff88 ):
            decorr_dir = os.path.join(os.getenv('DIA_OUT_DIR'), 'decorr')
            sci_pointing = sci_info['pointing']
            sci_sca = sci_info['sca']
            template_pointing = templ_info['pointing']
            template_sca = templ_info['sca']
            ra = objinfo['ra']
            dec = objinfo['dec']
            decorr_psf_path = os.path.join( decorr_dir,
                                            f'psf_{band}_{sci_pointing}_{sci_sca}_-_{band}_{template_pointing}_{template_sca}.fits' )
            decorr_zptimg_path = os.path.join( decorr_dir,
                                               f'stamp_{ra}_{dec}_decorr_zptimg_Roman_TDS_simple_model_{band}_{sci_pointing}_{sci_sca}.fits' )
            decorr_diff_path = os.path.join( decorr_dir,
                                             f'decor_diff_{band}_{sci_pointing}_{sci_sca}_-_{band}_{template_pointing}_{template_sca}.fits')

            for img, savepath, hdr in zip( [ sfftifier.PixA_DIFF_GPU, sfftifier.PixA_Ctarget_GPU, sfftifier.PSF_target_GPU ],
                                           [ decorr_diff_path,        decorr_zptimg_path,         decorr_psf_path ],
                                           [ sfftifier.hdr_target,    sfftifier.hdr_target,       None ] ):
                decorimg = sfftifier.apply_decorrelation( img )

                with nvtx.annotate( "writing FITS file", color=0x44ff88 ):
                    fits.writeto( savepath, cupy.asnumpy( decorimg ).T, header=hdr, overwrite=True )
        
# TODO -- stamp making (postprocess)

# TODO : make lightcurve

# logger.info( f"*** Running mk_lc for {sn} {band}" )
# with nvtx.annotate( "mk_lc", color="violet" ):
#     mk_lc.run( sn, band, n_templates=1, verbose=True, cpus_per_task=ncpus )

logger.info( f"*** Done with {sn} {band}" )
