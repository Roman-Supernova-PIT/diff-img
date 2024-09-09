import os
from phrosty.imagesubtraction import imsub

refpath = os.path.join( os.getenv('SIMS_DIR'), 'Roman_TDS_simple_model_R062_6_17.fits.gz' )
scipath = os.path.join( os.getenv('SIMS_DIR'), 'Roman_TDS_simple_model_R062_35083_8.fits.gz' )

s = imsub(20172782,'R062',os.getenv('DIA_OUT_DIR'))

s.set_template(refpath)
s.set_science(scipath)
s.preprocess_template()
s.preprocess_sci()
s.set_template_psf()
s.set_science_psf()
s.crossconv()
s.diff()
