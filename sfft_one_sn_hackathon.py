import nvtx

from astropy.io import fits

from phrosty.utils import set_logger


import preprocess
import sfftdiff
import postprocess
import mk_lc

sne= '20172782'
band = 'R062'
sci_list_path = "/phrosty_etal/20172782_instances_science.csv"
templ_list_path = "/phrosty_etal/20172782_instances_templates.csv"

ncpus = 1

logger = set_logger( "wrapper", "wrapper" )

objinfo, template_imgs, science_imgs = preprocess.get_transient_info_and_images( sn,
                                                                                 band,
                                                                                 sci_list_path,
                                                                                 templ_list_path )
preprocess.sky_sub_all_images( template_imgs, science_imgs, cpus_per_task=ncpus )
preprocess.get_psfs( objinfo, tempalte_imgs, science_imgs, cpus_per_tas=ncpus )

for templ_info in template_imgs.values():
    for sci_info in science_imgs.values():
        sfftifier = align_and_preconvolve( objinfo['ra'], objinfo['dec'], band, ( templ_info, sci_info ) )
        # Save and free
        
        sfftifier.sfft_subtract()
        # Save and free
        
        sfftifier.find_decorrelation()
        # Save and free

        decorr_dir = os.path.join(out_path, 'decorr')
        decorr_psf_path = os.path.join( decorr_dir,
                                        f'psf_{band}_{sci_pointing}_{sci_sca}_-_{band}_{template_pointing}_{template_sca}.fits' )
        decorr_zptimg_path = os.path.join( decorr_dir,
                                           f'stamp_{ra}_{dec}_decorr_zptimg_Roman_TDS_simple_model_{band}_{sci_pointing}_{sci_sca}.fits' )
        decorr_diff_path = os.path.join( decorr_dir,
                                         f'decor_diff_{band}_{sci_pointing}_{sci_sca}_-_{band}_{template_pointing}_{template_sca}.fits')
        
        for img, savepath, hdr in ( [ sfftifier.PixA_DIFF_GPU, sfftifier.PixA.Ctarget_GPU, sfftifier.PSF_target_GPU ],
                                    [ decorr_diff_path,        decorr_zptimg_path,         decorr_psf_path ],
                                    [ sfftifier.hdr_target,    sfftifier.hdr_target,       None ] ):
            decorimg = sfftifier.apply_decorrelation( self, img )

            # Try just saving the cupy array to fits to see if that works
            # If it doesn't, we're gonna have to asnumpy it 
            fits.writeto( savepath, decorimg.T, header=hdr )
        
# TODO -- stamp making (postprocess)

# TODO : make lightcurve

# logger.info( f"*** Running mk_lc for {sn} {band}" )
# with nvtx.annotate( "mk_lc", color="violet" ):
#     mk_lc.run( sn, band, n_templates=1, verbose=True, cpus_per_task=ncpus )

logger.info( f"*** Done with {sn} {band}" )
