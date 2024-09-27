import os
import re
import numpy as np
import os.path as pa
from astropy.io import fits
from regions import Regions
from skimage.draw import polygon2mask
from sfft.utils.ReadWCS import Read_WCS
from sfft.utils.pyAstroMatic.PYSEx import PY_SEx
from sfft.utils.SExSkySubtract import SEx_SkySubtract
MAINDIR = "/global/u1/l/leihu/JupyterLab/Hackathon/diff-img/sfftTestPurePack"   # FIXME

# setup reference & science filename
refname = 'Roman_WAS_simple_model_H158_9758_15'
sciname = 'Roman_WAS_simple_model_H158_11832_15'

# Note: please unpack .fz files manually
# * extract the science extension
FITS_REF0 = MAINDIR + '/preproc/input/%s.fits' %refname
FITS_SCI0 = MAINDIR + '/preproc/input/%s.fits' %sciname

for FITS_obj in [FITS_REF0, FITS_SCI0]:
    FITS_out = MAINDIR + '/preproc/output/%s.sciE.fits' %(pa.basename(FITS_obj)[:-5])
    with fits.open(FITS_obj) as hdl:
        hdr1 = hdl[1].header
        hdr1['GAIN'] = 1.0
        hdr1['SATURATE'] = 120000.  # trivial placeholder
        fits.HDUList([fits.PrimaryHDU(data=hdl[1].data, header=hdr1)])\
            .writeto(FITS_out, overwrite=True)

# * sky subtraction
FITS_REF1 = MAINDIR + '/preproc/output/%s.sciE.fits' %refname
FITS_SCI1 = MAINDIR + '/preproc/output/%s.sciE.fits' %sciname

for FITS_obj in [FITS_REF1, FITS_SCI1]:
    FITS_skysub = MAINDIR + '/input/%s.skysub.fits' %(pa.basename(FITS_obj)[:-5])
    SEx_SkySubtract.SSS(FITS_obj=FITS_obj, FITS_skysub=FITS_skysub, FITS_sky=None, FITS_skyrms=None, \
        SATUR_KEY='SATURATE', ESATUR_KEY='ESATUR', BACK_SIZE=64, BACK_FILTERSIZE=3, DETECT_THRESH=1.5, \
        DETECT_MINAREA=5, DETECT_MAXAREA=0, VERBOSE_LEVEL=2)

# * create detection masks 
FITS_REF2 = MAINDIR + '/input/%s.sciE.skysub.fits' %refname
FITS_SCI2 = MAINDIR + '/input/%s.sciE.skysub.fits' %sciname

SExParam = ['X_IMAGE', 'Y_IMAGE', 'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO', 'FLAGS', \
    'FLUX_RADIUS', 'FWHM_IMAGE', 'A_IMAGE', 'B_IMAGE', 'KRON_RADIUS', 'THETA_IMAGE', 'SNR_WIN']

PixA_SEG_REF = PY_SEx.PS(FITS_obj=FITS_REF2, SExParam=SExParam, GAIN_KEY='GAIN', SATUR_KEY='SATURATE', \
    BACK_TYPE='MANUAL', BACK_VALUE=0.0, BACK_SIZE=64, BACK_FILTERSIZE=3, DETECT_THRESH=1.5, \
    DETECT_MINAREA=5, DETECT_MAXAREA=0, DEBLEND_MINCONT=0.001, BACKPHOTO_TYPE='LOCAL', \
    CHECKIMAGE_TYPE='SEGMENTATION', AddRD=True, ONLY_FLAGS=None, XBoundary=0.0, YBoundary=0.0, \
    MDIR=None, VERBOSE_LEVEL=1)[1][0]

PixA_SEG_SCI = PY_SEx.PS(FITS_obj=FITS_SCI2, SExParam=SExParam, GAIN_KEY='GAIN', SATUR_KEY='SATURATE', \
    BACK_TYPE='MANUAL', BACK_VALUE=0.0, BACK_SIZE=64, BACK_FILTERSIZE=3, DETECT_THRESH=1.5, \
    DETECT_MINAREA=5, DETECT_MAXAREA=0, DEBLEND_MINCONT=0.001, BACKPHOTO_TYPE='LOCAL', \
    CHECKIMAGE_TYPE='SEGMENTATION', AddRD=True, ONLY_FLAGS=None, XBoundary=0.0, YBoundary=0.0, \
    MDIR=None, VERBOSE_LEVEL=1)[1][0]

def create_banmask(FITS_obj, BANREG_POLYGON):
    _phr = fits.getheader(FITS_obj, ext=0)
    N0, N1 = _phr["NAXIS1"], _phr["NAXIS2"]
    w = Read_WCS.RW(hdr=_phr, VERBOSE_LEVEL=1)
    BANMASK_POLYGON = np.zeros((N0, N1)).astype(bool)
    regions = Regions.read(BANREG_POLYGON, format='ds9')
    for reg in regions:
        _X, _Y = reg.vertices.to_pixel(w)
        polygon = np.array([_X, _Y]).T
        mask = polygon2mask((N0, N1), polygon)
        BANMASK_POLYGON[mask] = True
    return BANMASK_POLYGON

REFINE_BANREG_POLYGON = MAINDIR + '/preproc/input/brightstars.reg'
BANMASK_POLYGON_REF = create_banmask(FITS_obj=FITS_REF2, BANREG_POLYGON=REFINE_BANREG_POLYGON)
BANMASK_POLYGON_SCI = create_banmask(FITS_obj=FITS_SCI2, BANREG_POLYGON=REFINE_BANREG_POLYGON)

FITS_DETMASK_REF = MAINDIR + '/input/%s.sciE.skysub.detmask.fits' %refname
PixA_DETMASK_REF = np.logical_and(PixA_SEG_REF > 0, ~BANMASK_POLYGON_REF)
with fits.open(FITS_REF2) as hdl:
    hdl[0].data[:, :] = PixA_DETMASK_REF.astype(float).T
    hdl.writeto(FITS_DETMASK_REF, overwrite=True)

FITS_DETMASK_SCI = MAINDIR + '/input/%s.sciE.skysub.detmask.fits' %sciname
PixA_DETMASK_SCI = np.logical_and(PixA_SEG_SCI > 0, ~BANMASK_POLYGON_SCI)
with fits.open(FITS_SCI2) as hdl:
    hdl[0].data[:, :] = PixA_DETMASK_SCI.astype(float).T
    hdl.writeto(FITS_DETMASK_SCI, overwrite=True)