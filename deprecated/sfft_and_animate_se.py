from phrosty.imagesubtraction import imsub

refpath = '/cwork/mat90/RomanDESC_sims_2024/RomanTDS/images/simple_model/R062/6/Roman_TDS_simple_model_R062_6_17.fits.gz'
scipath = '/cwork/mat90/RomanDESC_sims_2024/RomanTDS/images/simple_model/R062/35083/Roman_TDS_simple_model_R062_35083_8.fits.gz'

s = imsub(20172782,'R062','/work/lna18/imsub_out')
s.set_template(refpath)
s.set_science(scipath)
s.preprocess_template()
s.preprocess_sci()
s.set_template_psf()
s.set_science_psf()
s.crossconv()
s.diff()