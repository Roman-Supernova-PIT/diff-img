import os
import sfft

#!/usr/bin/env python
# coding: utf-8

# In[13]:


# setup directory paths
# MAINDIR = '/run/mount/nas/Public/Thomas/RMSIM/ship2pitsn'   # FIXME
input_dir = os.getenv( 'SIMS_DIR' )
output_dir = os.getenv( 'DIA_OUT_DIR' )

# setup reference & science filename
refname = 'Roman_TDS_simple_model_R062_6_17'
sciname = 'Roman_TDS_simple_model_R062_35083_8'


# ### Preproc step1. extract science extension from input FITS

# In[5]:


import re
import os.path as pa
from astropy.io import fits

# TODO: please funpack the .fz files in the directory or get the two input fits files from Lauren.
FITS_REF0 = input_dir + '/%s.fits' %refname
FITS_SCI0 = input_dir + '/%s.fits' %sciname

for FITS_obj in [FITS_REF0, FITS_SCI0]:
    FITS_out = output_dir + '/%s.sciE.fits' %(pa.basename(FITS_obj)[:-5])
    with fits.open(FITS_obj) as hdl:
        hdr1 = hdl[1].header
        hdr1['GAIN'] = 1.0
        hdr1['SATURATE'] = 120000.  # trivial placeholder
        fits.HDUList([fits.PrimaryHDU(data=hdl[1].data, header=hdr1)])\
            .writeto(FITS_out, overwrite=True)


# ### Preproc step2. run sky subtraction using SourceExtractor 
# #### NOTE: this step will be optional in future version

# In[9]:


import os
import numpy as np
from astropy.io import fits
from sfft.utils.SExSkySubtract import SEx_SkySubtract

FITS_REF1 = output_dir + '/%s.sciE.fits' %refname
FITS_SCI1 = output_dir + '/%s.sciE.fits' %sciname

for FITS_obj in [FITS_REF1, FITS_SCI1]:
    FITS_skysub = output_dir + '/%s.skysub.fits' %(pa.basename(FITS_obj)[:-5])
    SEx_SkySubtract.SSS(FITS_obj=FITS_obj, FITS_skysub=FITS_skysub, FITS_sky=None, FITS_skyrms=None, \
        SATUR_KEY='SATURATE', ESATUR_KEY='ESATUR', BACK_SIZE=64, BACK_FILTERSIZE=3, DETECT_THRESH=1.5, \
        DETECT_MINAREA=5, DETECT_MAXAREA=0, VERBOSE_LEVEL=2)


# ### Preproc step3. align science image to the reference image frame using SWarp
# #### NOTE: To avoid negative artifacts, here we use BILINEAR interpolation function for resampling.
# #### NOTE: The downside is that the interpolation will enlarge the psf fwhm and will introduce a small flux bias (~1%).
# #### FIXME: Better image alignment methods may be available when the reference and science images are mosaic images.

# In[17]:


from sfft.utils.pyAstroMatic.PYSWarp import PY_SWarp

FITS_REF2 = output_dir + '/%s.sciE.skysub.fits' %refname
FITS_SCI2 = output_dir + '/%s.sciE.skysub.fits' %sciname
FITS_SCI3 = FITS_SCI2[:-5] + '.resamp.fits'

PY_SWarp.PS(FITS_obj=FITS_SCI2, FITS_ref=FITS_REF2, FITS_resamp=FITS_SCI3, \
    GAIN_KEY='GAIN', SATUR_KEY='SATURATE', OVERSAMPLING=1, RESAMPLING_TYPE='BILINEAR', \
    SUBTRACT_BACK='N', FILL_VALUE=np.nan, VERBOSE_TYPE='NORMAL', VERBOSE_LEVEL=1)


# ### Preproc step4. retrieve psf model using roman_imsim
# #### FIXME: please adjust PATH in was.yaml: [input: obseq_data: file_name: /your/path/Roman_WAS_obseq_11_1_23.fits]

# In[9]:


import sys
import math
import galsim
import os.path as pa
from astropy.io import fits
from roman_imsim.utils import roman_utils

# assert pa.exists(config_file)

# # retrieve psf model at reference image center
# util_ref = roman_utils(config_file=config_file, image_name='%s.fits' %refname)
# psf_image_ref = util_ref.getPSF_Image(501, x=2048.5, y=2048.5).array

# FITS_PSF_REF = output_dir + '/%s.centPSF.fits' %refname
# fits.HDUList([fits.PrimaryHDU(data=psf_image_ref, header=None)]).writeto(FITS_PSF_REF, overwrite=True)

# # retrieve psf model at (unresampled) science image center
# util_sci = roman_utils(config_file=config_file, image_name='%s.fits' %sciname)
# psf_image_sci = util_ref.getPSF_Image(501, x=2048.5, y=2048.5).array

# FITS_PSF_SCI = output_dir + '/%s.centPSF.fits' %sciname
# fits.HDUList([fits.PrimaryHDU(data=psf_image_sci, header=None)]).writeto(FITS_PSF_SCI, overwrite=True)

FITS_PSF_SCI = os.path.join( os.getenv("SIMS_DIR"), "psf_7.551093401915147_-44.80718106491529_R062_35083_8.fits" )
FITS_PSF_REF = os.path.join( os.getenv("SIMS_DIR"), "psf_7.551093401915147_-44.80718106491529_R062_6_17.fits" )

# ### Preproc step5. rotate science psf model accroding to the image alignment

# In[17]:


import sys
import numpy as np
from sfft.utils.ReadWCS import Read_WCS
from sfft.utils.ImageZoomRotate import Image_ZoomRotate

# Step 3p: rotate PSF model to align resampled image
def calculate_skyN_vector(wcshdr, x_start, y_start, shift_dec=1.0):
    w = Read_WCS.RW(wcshdr, VERBOSE_LEVEL=1)
    ra_start, dec_start = w.all_pix2world(np.array([[x_start, y_start]]), 1)[0]
    ra_end, dec_end = ra_start, dec_start + shift_dec/3600.0
    x_end, y_end = w.all_world2pix(np.array([[ra_end, dec_end]]), 1)[0]
    skyN_vector = np.array([x_end - x_start, y_end - y_start])
    return skyN_vector

def calculate_rotate_angle(vector_ref, vector_obj):
    rad = np.arctan2(np.cross(vector_ref, vector_obj), np.dot(vector_ref, vector_obj))
    rotate_angle = np.rad2deg(rad)
    if rotate_angle < 0.0: rotate_angle += 360.0 
    return rotate_angle

# calculate rotation angle during resampling
FITS_SCI2 = output_dir + '/%s.sciE.skysub.fits' %sciname
FITS_SCI3 = FITS_SCI2[:-5] + '.resamp.fits'

_phdr = fits.getheader(FITS_SCI2, ext=0)
_w = Read_WCS.RW(_phdr, VERBOSE_LEVEL=1)
x0, y0 = 0.5 + int(_phdr['NAXIS1'])/2.0, 0.5 + int(_phdr['NAXIS2'])/2.0
ra0, dec0 = _w.all_pix2world(np.array([[x0, y0]]), 1)[0]
skyN_vector = calculate_skyN_vector(wcshdr=_phdr, x_start=x0, y_start=y0)

_phdr = fits.getheader(FITS_SCI3, ext=0)
_w = Read_WCS.RW(_phdr, VERBOSE_LEVEL=1)
x1, y1 = _w.all_world2pix(np.array([[ra0, dec0]]), 1)[0]
skyN_vectorp = calculate_skyN_vector(wcshdr=_phdr, x_start=x1, y_start=y1)
PATTERN_ROTATE_ANGLE = calculate_rotate_angle(vector_ref=skyN_vector, vector_obj=skyN_vectorp)

# perform rotation to get rotated psf model for resampled science
#FITS_PSF_SCI = output_dir + '/%s.centPSF.fits' %sciname
PixA_PSF_SCI = fits.getdata(FITS_PSF_SCI, ext=0).T
PSF_PSF_ReSCI = Image_ZoomRotate.IZR(PixA_obj=PixA_PSF_SCI, ZOOM_SCALE_X=1., \
    ZOOM_SCALE_Y=1., PATTERN_ROTATE_ANGLE=PATTERN_ROTATE_ANGLE, \
    RESAMPLING_TYPE='BILINEAR', FILL_VALUE=0.0, VERBOSE_LEVEL=1)[0]

FITS_PSF_ReSCI = output_dir + '/%s.resamp.centPSF.fits' %sciname
fits.HDUList([fits.PrimaryHDU(data=PSF_PSF_ReSCI.T, header=None)]).writeto(FITS_PSF_ReSCI, overwrite=True)


# ### Preproc step6. perform cross convolution for coarsely psf aligment (ref * psf_sci and sci * psf_ref)

# In[18]:


from astropy.convolution import convolve_fft

FITS_lREF = output_dir + '/%s.sciE.skysub.fits' %refname
FITS_lSCI = output_dir + '/%s.sciE.skysub.resamp.fits' %sciname

FITS_PSF_lREF = FITS_PSF_REF # output_dir + '/%s.fits' %refname
FITS_PSF_lSCI = output_dir + '/%s.resamp.centPSF.fits' %sciname

PixA_lREF = fits.getdata(FITS_lREF, ext=0).T
PixA_lSCI = fits.getdata(FITS_lSCI, ext=0).T
PixA_PSF_lREF = fits.getdata(FITS_PSF_lREF, ext=0).T
PixA_PSF_lSCI = fits.getdata(FITS_PSF_lSCI, ext=0).T

# convolve (resampled) science psf on reference
PixA_lREF_convd = convolve_fft(PixA_lREF, PixA_PSF_lSCI, boundary='fill', \
    nan_treatment='fill', fill_value=0.0, normalize_kernel=True)
FITS_lREF_convd = FITS_lREF[:-5] + '.crossConvd.fits'

if FITS_lREF_convd is not None:
    with fits.open(FITS_lREF) as hdl:
        _message = 'Convolving image ... \n # %s' %FITS_lREF_convd
        print('\nMeLOn CheckPoint: %s!' %_message)
        hdl[0].data[:, :] = PixA_lREF_convd.T
        hdl.writeto(FITS_lREF_convd, overwrite=True)
    
# convolve reference psf on (resampled) science
PixA_lSCI_convd = convolve_fft(PixA_lSCI, PixA_PSF_lREF, boundary='fill', \
    nan_treatment='fill', fill_value=0.0, normalize_kernel=True)

FITS_lSCI_convd = FITS_lSCI[:-5] + '.crossConvd.fits'
if FITS_lSCI_convd is not None:
    with fits.open(FITS_lSCI) as hdl:
        _message = 'Convolving image ... \n # %s' %FITS_lSCI_convd
        print('\nMeLOn CheckPoint: %s!' %_message)
        hdl[0].data[:, :] = PixA_lSCI_convd.T
        hdl.writeto(FITS_lSCI_convd, overwrite=True)


# ### Preproc step7. make a 1K cutout to perform sfft subtraction
# #### NOTE: the full frame subtraction is not optimized yet, let's test a cutout for now.

# In[19]:


from sfft.utils.StampGenerator import Stamp_Generator

# Step temp: make a cutout for subtraction test
FITS_lREF = output_dir + '/%s.sciE.skysub.fits' %refname
FITS_lSCI = output_dir + '/%s.sciE.skysub.resamp.fits' %sciname

FITS_REF = FITS_lREF[:-5] + '.crossConvd.fits'
FITS_SCI = FITS_lSCI[:-5] + '.crossConvd.fits'

STAMP_IMGSIZE = (1024, 1024) # stamp image size
COORD = np.array([[1195.5299, 2972.6193]])  # image coordinate

for FITS_obj in [FITS_lREF, FITS_lSCI, FITS_REF, FITS_SCI]:
    FITS_StpLst = [FITS_obj[:-5] + '.stamp.fits']
    Stamp_Generator.SG(FITS_obj=FITS_obj, COORD=COORD, COORD_TYPE='IMAGE', \
        STAMP_IMGSIZE=STAMP_IMGSIZE, FILL_VALUE=np.nan, FITS_StpLst=FITS_StpLst, VERBOSE_LEVEL=2)
    


# ### Preproc step7. create detection mask 

# In[22]:


import numpy as np
from astropy.io import fits
from sfft.utils.pyAstroMatic.PYSEx import PY_SEx

# Step 5. Run SFFT Subtraction
FITS_lREF = output_dir + '/%s.sciE.skysub.stamp.fits' %refname # use stamp
FITS_lSCI = output_dir + '/%s.sciE.skysub.resamp.stamp.fits' %sciname # use stamp
FITS_REF = output_dir + '/%s.sciE.skysub.crossConvd.stamp.fits'  %refname # use stamp
FITS_SCI = output_dir + '/%s.sciE.skysub.resamp.crossConvd.stamp.fits'  %sciname # use stamp

SExParam = ['X_IMAGE', 'Y_IMAGE', 'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO', 'FLAGS', \
    'FLUX_RADIUS', 'FWHM_IMAGE', 'A_IMAGE', 'B_IMAGE', 'KRON_RADIUS', 'THETA_IMAGE', 'SNR_WIN']

PixA_SEG_REF = PY_SEx.PS(FITS_obj=FITS_REF, SExParam=SExParam, GAIN_KEY='GAIN', SATUR_KEY='SATURATE', \
    BACK_TYPE='MANUAL', BACK_VALUE=0.0, BACK_SIZE=64, BACK_FILTERSIZE=3, DETECT_THRESH=1.5, \
    DETECT_MINAREA=5, DETECT_MAXAREA=0, DEBLEND_MINCONT=0.001, BACKPHOTO_TYPE='LOCAL', \
    CHECKIMAGE_TYPE='SEGMENTATION', AddRD=True, ONLY_FLAGS=None, XBoundary=0.0, YBoundary=0.0, \
    MDIR=None, VERBOSE_LEVEL=1)[1][0]

PixA_SEG_SCI = PY_SEx.PS(FITS_obj=FITS_SCI, SExParam=SExParam, GAIN_KEY='GAIN', SATUR_KEY='SATURATE', \
    BACK_TYPE='MANUAL', BACK_VALUE=0.0, BACK_SIZE=64, BACK_FILTERSIZE=3, DETECT_THRESH=1.5, \
    DETECT_MINAREA=5, DETECT_MAXAREA=0, DEBLEND_MINCONT=0.001, BACKPHOTO_TYPE='LOCAL', \
    CHECKIMAGE_TYPE='SEGMENTATION', AddRD=True, ONLY_FLAGS=None, XBoundary=0.0, YBoundary=0.0, \
    MDIR=None, VERBOSE_LEVEL=1)[1][0]

LYMASK_BKG = np.logical_and(PixA_SEG_REF == 0, PixA_SEG_SCI == 0)   # background-mask
LYMASK_RDET = ~LYMASK_BKG  # detection-mask


# ### SFFT Subtraction (public Github version): run polynominal form SFFT
# #### NOTE: I found that B-spline form SFFT (https://arxiv.org/abs/2309.09143) can have slightly better subtraction performance on roman sim. 
# #### NOTE: However, the B-spline SFFT code has not yet incorpated into the public software. It would be available in the next version of SFFT (https://github.com/thomasvrussell/sfft). 

# In[43]:


import os
import os.path as pa
from tempfile import mkdtemp
from sfft.CustomizedPacket import Customized_Packet
from sfft.utils.SkyLevelEstimator import SkyLevel_Estimator
from sfft.utils.SFFTSolutionReader import Realize_MatchingKernel
from sfft.utils.DeCorrelationCalculator import DeCorrelation_Calculator

FITS_PSF_lREF = FITS_PSF_REF #  '/%s.centPSF.fits' %refname
FITS_PSF_lSCI = output_dir + '/%s.resamp.centPSF.fits' %sciname

FITS_lREF = output_dir + '/%s.sciE.skysub.stamp.fits' %refname # use stamp
FITS_lSCI = output_dir + '/%s.sciE.skysub.resamp.stamp.fits' %sciname # use stamp

FITS_REF = output_dir + '/%s.sciE.skysub.crossConvd.stamp.fits'  %refname # use stamp
FITS_SCI = output_dir + '/%s.sciE.skysub.resamp.crossConvd.stamp.fits'  %sciname # use stamp

FITS_DIFF = FITS_SCI[:-5] + '.polysfftdiff.fits'
FITS_Solution = FITS_SCI[:-5] + '.polysfftsolution.fits'
FITS_DCDIFF = FITS_SCI[:-5] + '.polysfftdiff.DeCorrelated.fits'

# zero-out background to create masked image pair 
TDIR = mkdtemp(suffix=None, prefix='mask', dir=None)
FITS_mREF = TDIR + '%s.masked.fits' %(pa.basename(FITS_REF)[:-5])
FITS_mSCI = TDIR + '%s.masked.fits' %(pa.basename(FITS_SCI)[:-5])

with fits.open(FITS_REF) as hdl:
    _PixA = hdl[0].data.T
    _PixA[LYMASK_BKG] = 0.0
    hdl[0].data[:, :] = _PixA.T
    hdl.writeto(FITS_mREF, overwrite=True)

with fits.open(FITS_SCI) as hdl:
    _PixA = hdl[0].data.T
    _PixA[LYMASK_BKG] = 0.0
    hdl[0].data[:, :] = _PixA.T
    hdl.writeto(FITS_mSCI, overwrite=True)

# configuration for sfft subtraction
ForceConv = 'REF'       # convolve which side, 'SCI' or 'REF'
GKerHW = 9
KerPolyOrder = 3        # polynomial degree for matching kerenl spatial variation 
BGPolyOrder = 0         # polynomial degree for differential background variation, trivial here
ConstPhotRatio = True   # constant flux scaling?

BACKEND_4SUBTRACT = 'Cupy'      # FIXME: Please use 'Numpy' if no gpu device avaiable
CUDA_DEVICE_4SUBTRACT = '0'     # gpu device index, only for Cupy backend
NUM_CPU_THREADS_4SUBTRACT = 8   # number of cpu threads, only for Numpy backend

# run polynomial form sfft subtraction 
Customized_Packet.CP(FITS_REF=FITS_REF, FITS_SCI=FITS_SCI, FITS_mREF=FITS_mREF, FITS_mSCI=FITS_mSCI, \
    ForceConv=ForceConv, GKerHW=GKerHW, FITS_DIFF=FITS_DIFF, FITS_Solution=FITS_Solution, \
    KerPolyOrder=KerPolyOrder, BGPolyOrder=BGPolyOrder, ConstPhotRatio=ConstPhotRatio, \
    BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, CUDA_DEVICE_4SUBTRACT=CUDA_DEVICE_4SUBTRACT, \
    NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT)
os.system('rm -rf %s' %TDIR)

# run noise decorrelation for polynomial form sfft subtraction 
PixA_PSF_lREF = fits.getdata(FITS_PSF_lREF, ext=0).T
PixA_PSF_lSCI = fits.getdata(FITS_PSF_lSCI, ext=0).T

PixA_lREF = fits.getdata(FITS_lREF, ext=0).T # use stamp
PixA_lSCI = fits.getdata(FITS_lSCI, ext=0).T # use stamp
PixA_DIFF = fits.getdata(FITS_DIFF, ext=0).T 

bkgsig_REF = SkyLevel_Estimator.SLE(PixA_obj=PixA_lREF)[1]
bkgsig_SCI = SkyLevel_Estimator.SLE(PixA_obj=PixA_lSCI)[1]

N0, N1 = PixA_lSCI.shape
XY_q = np.array([[N0/2.+0.5, N1/2.+0.5]])
MKerStack = Realize_MatchingKernel(XY_q).FromFITS(FITS_Solution=FITS_Solution)
MK_Fin = MKerStack[0]

DCKer = DeCorrelation_Calculator.DCC(MK_JLst=[PixA_PSF_lREF], SkySig_JLst=[bkgsig_SCI], \
    MK_ILst=[PixA_PSF_lSCI], SkySig_ILst=[bkgsig_REF], MK_Fin=MK_Fin, \
    KERatio=2.0, VERBOSE_LEVEL=2)

PixA_DCDIFF = convolve_fft(PixA_DIFF, DCKer, boundary='fill', \
    nan_treatment='fill', fill_value=0.0, normalize_kernel=True)

with fits.open(FITS_DIFF) as hdl:
    hdl[0].data[:, :] = PixA_DCDIFF.T
    hdl.writeto(FITS_DCDIFF, overwrite=True)

# roughly estimate the SNR map for the decorrelated difference image
# WARNING: the noise propagation is highly simplified.

GAIN = 1.0  
PixA_varREF = np.clip(PixA_lREF/GAIN, a_min=0.0, a_max=None) + bkgsig_REF**2
PixA_varSCI = np.clip(PixA_lSCI/GAIN, a_min=0.0, a_max=None) + bkgsig_SCI**2
PixA_DIFF_Noise = np.sqrt(PixA_varREF + PixA_varSCI)

FITS_DSNR = FITS_DIFF[:-5] + '.DeCorrelated.SNR.fits'
PixA_DSNR = PixA_DCDIFF/PixA_DIFF_Noise
with fits.open(FITS_DIFF) as hdl:
    hdl[0].data[:, :] = PixA_DSNR.T
    hdl.writeto(FITS_DSNR, overwrite=True)


# In[ ]:




