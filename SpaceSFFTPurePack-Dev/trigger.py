import os
import sys
import nvtx
import cupy as cp
import numpy as np
import os.path as pa
from astropy.io import fits
GDIR = "/hildafs/projects/phy220048p/leihu/AstroWork/RomanSNPIT/SpaceSFFTPurePack-OPT"
sys.path.insert(1, GDIR + "/scripts")
from SpaceSFFTFlow import SpaceSFFT_CupyFlow_NVTX

input_dir = GDIR + '/input'
output_dir = GDIR + '/output'

# setup reference & science filename
refname = 'Roman_WAS_simple_model_H158_9758_15'
sciname = 'Roman_WAS_simple_model_H158_11832_15'

FITS_REF = input_dir + "/%s.sciE.skysub.fits" % refname
FITS_oSCI = input_dir + "/%s.sciE.skysub.fits" % sciname

FITS_REF_DMASK = input_dir + "/%s.sciE.skysub.detmask.fits" % refname
FITS_oSCI_DMASK = input_dir + "/%s.sciE.skysub.detmask.fits" % sciname

FITS_PSF_REF = input_dir + '/%s.centPSF.fits' %refname
FITS_PSF_oSCI = input_dir + '/%s.centPSF.fits' %sciname

assert pa.exists(FITS_REF) and pa.exists(FITS_oSCI) 
assert pa.exists(FITS_REF_DMASK) and pa.exists(FITS_oSCI_DMASK)

# read inputs
hdr_REF = fits.getheader(FITS_REF, ext=0)
hdr_oSCI = fits.getheader(FITS_oSCI, ext=0)

PixA_REF = fits.getdata(FITS_REF, ext=0).T
PixA_oSCI = fits.getdata(FITS_oSCI, ext=0).T

PixA_REF_DMASK = fits.getdata(FITS_REF_DMASK, ext=0).T
PixA_oSCI_DMASK = fits.getdata(FITS_oSCI_DMASK, ext=0).T

PSF_REF = fits.getdata(FITS_PSF_REF, ext=0).T
PSF_oSCI = fits.getdata(FITS_PSF_oSCI, ext=0).T

# real run
for i in range(2):
    with nvtx.annotate("toGPU", color="#714CF9"):
        if not PixA_REF.flags['C_CONTIGUOUS']:
            PixA_REF = np.ascontiguousarray(PixA_REF, np.float64)
            PixA_REF_GPU = cp.array(PixA_REF)
        else: PixA_REF_GPU = cp.array(PixA_REF.astype(np.float64))

        if not PixA_oSCI.flags['C_CONTIGUOUS']:
            PixA_oSCI = np.ascontiguousarray(PixA_oSCI, np.float64)
            PixA_oSCI_GPU = cp.array(PixA_oSCI)
        else: PixA_oSCI_GPU = cp.array(PixA_oSCI.astype(np.float64))

        if not PixA_REF_DMASK.flags['C_CONTIGUOUS']:
            PixA_REF_DMASK = np.ascontiguousarray(PixA_REF_DMASK, np.float64)
            PixA_REF_DMASK_GPU = cp.array(PixA_REF_DMASK)
        else: PixA_REF_DMASK_GPU = cp.array(PixA_REF_DMASK.astype(np.float64))

        if not PixA_oSCI_DMASK.flags['C_CONTIGUOUS']:
            PixA_oSCI_DMASK = np.ascontiguousarray(PixA_oSCI_DMASK, np.float64)
            PixA_oSCI_DMASK_GPU = cp.array(PixA_oSCI_DMASK)
        else: PixA_oSCI_DMASK_GPU = cp.array(PixA_oSCI_DMASK.astype(np.float64))

        if not PSF_REF.flags['C_CONTIGUOUS']:
            PSF_REF = np.ascontiguousarray(PSF_REF, np.float64)
            PSF_REF_GPU = cp.array(PSF_REF)
        else: PSF_REF_GPU = cp.array(PSF_REF.astype(np.float64))

        if not PSF_oSCI.flags['C_CONTIGUOUS']:
            PSF_oSCI = np.ascontiguousarray(PSF_oSCI, np.float64)
            PSF_oSCI_GPU = cp.array(PSF_oSCI)
        else: PSF_oSCI_GPU = cp.array(PSF_oSCI.astype(np.float64))

    PixA_DIFF_GPU, PixA_DCDIFF_GPU, PixA_DSNR_GPU = SpaceSFFT_CupyFlow_NVTX.SSCFN(hdr_REF=hdr_REF, hdr_oSCI=hdr_oSCI, 
        PixA_REF_GPU=PixA_REF_GPU, PixA_oSCI_GPU=PixA_oSCI_GPU, PixA_REF_DMASK_GPU=PixA_REF_DMASK_GPU, 
        PixA_oSCI_DMASK_GPU=PixA_oSCI_DMASK_GPU, PSF_REF_GPU=PSF_REF_GPU, PSF_oSCI_GPU=PSF_oSCI_GPU, 
        GKerHW=9, KerPolyOrder=2, BGPolyOrder=0, ConstPhotRatio=True, CUDA_DEVICE_4SUBTRACT='0', GAIN=1.0)

    with nvtx.annotate("toCPU", color="#FD89B4"):
        PixA_DCDIFF = cp.asnumpy(PixA_DCDIFF_GPU)
        PixA_DSNR = cp.asnumpy(PixA_DSNR_GPU)

    if i == 0:
        # save the final results
        FITS_DCDIFF = output_dir + "/%s.sciE.skysub.resamp.sfftdiff.decorr.fits" % sciname
        with fits.open(FITS_REF) as hdul:
            hdul[0].data = PixA_DCDIFF.T
            hdul.writeto(FITS_DCDIFF, overwrite=True)

        FITS_DSNR = output_dir + "/%s.sciE.skysub.resamp.sfftdiff.decorr.snr.fits" % sciname
        with fits.open(FITS_REF) as hdul:
            hdul[0].data = PixA_DSNR.T
            hdul.writeto(FITS_DSNR, overwrite=True)
    print("Done!")
