import os
import sys
import math
import nvtx
import galsim
import cupy as cp
import numpy as np
import os.path as pa
from astropy.io import fits
from tempfile import mkdtemp
from sfft.utils.ReadWCS import Read_WCS
from roman_imsim.utils import roman_utils
from astropy.convolution import convolve_fft
from sfft.CustomizedPacket import Customized_Packet
from sfft.utils.StampGenerator import Stamp_Generator
from sfft.utils.SkyLevelEstimator import SkyLevel_Estimator
from sfft.utils.SFFTSolutionReader import Realize_MatchingKernel
from sfft.utils.DeCorrelationCalculator import DeCorrelation_Calculator
# GDIR = '/global/u1/l/leihu/JupyterLab/Hackathon/diff-img/sfftTestPurePack'  # nersc
GDIR = '/hack_team_01/leiworkflow/diff-img/sfftTestPurePack'  # curoisity

utils_dir = GDIR + '/utils'
sys.path.insert(1, utils_dir)
from ImageZoomRotate import Image_ZoomRotate
from CudaResampling import Cuda_Resampling

aux_dir = GDIR + '/auxiliary'
config_file = aux_dir + '/was.yaml'
assert pa.exists(config_file)

# *** Example of Image Subtraction on Roman observations with SFFT 
# Author: Lei Hu <leihu@andrew.cmu.edu>  
# Last Verified to run: 2024-09-18

input_dir = GDIR + '/input'
output_dir = GDIR + '/output'

# setup reference & science filename
refname = 'Roman_WAS_simple_model_H158_9758_15'
sciname = 'Roman_WAS_simple_model_H158_11832_15'

def convolve_fft_cuda(PixA_Inp_GPU, KERNEL_GPU, PAD_FILL_VALUE=0., NORMALIZE_KERNEL=False, FFT_BACKEND="cupy"):
    # Warning: make sure no NaN values in PixA_Inp_GPU.
    
    N0, N1 = PixA_Inp_GPU.shape
    L0, L1 = KERNEL_GPU.shape
    assert L0 % 2 == 1 and L1 % 2 == 1

    W0, W1 = (L0 - 1)//2, (L1 - 1)//2 
    EN0, EN1 = N0 + 2*W0, N1 + 2*W1

    # zero padding on input image
    PixA_EInp_GPU = cp.pad(
        array=PixA_Inp_GPU, 
        pad_width=((W0, W0), (W1, W1)), 
        mode='constant', 
        constant_values=PAD_FILL_VALUE
    )
    
    # tail zero padding & cicular shift the kernel
    if NORMALIZE_KERNEL:
        KERNEL_TZP_GPU = cp.pad(KERNEL_GPU / cp.sum(KERNEL_GPU), \
            pad_width=((0, EN0 - L0), (0, EN1 - L1)), mode='constant', constant_values=0.)
    else:
        KERNEL_TZP_GPU = cp.pad(KERNEL_GPU, pad_width=((0, EN0 - L0), (0, EN1 - L1)), \
            mode='constant', constant_values=0.)
    KIMG_CSZ_GPU = cp.roll(cp.roll(KERNEL_TZP_GPU, -W0, axis=0), -W1, axis=1)
    
    # perform convolution in Fourier domain
    if FFT_BACKEND == "cupy":
        PixA_Out_GPU = cp.fft.ifft2(
            cp.fft.fft2(PixA_EInp_GPU) * cp.fft.fft2(KIMG_CSZ_GPU)
        ).real[W0: -W0, W1: -W1]
    
    # NOT fully tested yet
    # if FFT_BACKEND == "nvmath":
    #     PixA_Out_GPU = nvmath.fft.irfft(
    #         nvmath.fft.rfft(PixA_EInp_GPU) * nvmath.fft.rfft(KIMG_CSZ_GPU)
    #     )[W0: -W0, W1: -W1]
    
    return PixA_Out_GPU

def WorkFlow(refname, sciname, input_dir, output_dir, aux_dir, config_file):

    # *** step0a. prepare mapping for resampling [science image & detection mask onto reference frame]
    FITS_REF = input_dir + "/%s.sciE.skysub.fits" % refname
    FITS_oSCI = input_dir + "/%s.sciE.skysub.fits" % sciname

    FITS_REF_DMASK = input_dir + "/%s.sciE.skysub.detmask.fits" % refname
    FITS_oSCI_DMASK = input_dir + "/%s.sciE.skysub.detmask.fits" % sciname

    assert pa.exists(FITS_REF) and pa.exists(FITS_oSCI) 
    assert pa.exists(FITS_REF_DMASK) and pa.exists(FITS_oSCI_DMASK)

    FITS_SCI = output_dir + "/%s.sciE.skysub.resamp.fits" % sciname
    FITS_SCI_DMASK = output_dir + "/%s.sciE.skysub.detmask.resamp.fits" % sciname

    # create the mapping for object to target 
    CR1 = Cuda_Resampling(FITS_obj=FITS_oSCI, FITS_targ=FITS_REF, METHOD='BILINEAR', VERBOSE_LEVEL=1)
    CR2 = Cuda_Resampling(FITS_obj=FITS_oSCI_DMASK, FITS_targ=FITS_REF_DMASK, METHOD='BILINEAR', VERBOSE_LEVEL=1)
    
    # todo: the two mappings are same
    PixA_Eobj1, MappingDICT1 = CR1.mapping()
    PixA_Eobj2, MappingDICT2 = CR2.mapping()

    # *** step0b. resampling [science image & detection mask onto reference frame]
    PixA_resamp1 = CR1.resampling(PixA_Eobj=PixA_Eobj1, MappingDICT=MappingDICT1)
    PixA_resamp2 = CR2.resampling(PixA_Eobj=PixA_Eobj2, MappingDICT=MappingDICT2)

    with fits.open(FITS_REF) as hdl:
        PixA_resamp1[PixA_resamp1 == 0.] = np.nan
        hdl[0].data[:, :] = PixA_resamp1.T
        hdl.writeto(FITS_SCI, overwrite=True)

    with fits.open(FITS_REF_DMASK) as hdl:
        hdl[0].data[:, :] = PixA_resamp2.T
        hdl.writeto(FITS_SCI_DMASK, overwrite=True)

    # *** step1a. extract psf models
    # retrieve psf model at reference image center
    util_ref = roman_utils(config_file=config_file, image_name='%s.fits' %refname)
    psf_image_ref = util_ref.getPSF_Image(501, x=2048.5, y=2048.5).array

    FITS_PSF_REF = output_dir + '/%s.centPSF.fits' %refname
    fits.HDUList([fits.PrimaryHDU(data=psf_image_ref, header=None)]).writeto(FITS_PSF_REF, overwrite=True)

    # retrieve psf model at (unresampled) science image center
    util_sci = roman_utils(config_file=config_file, image_name='%s.fits' %sciname)
    psf_image_sci = util_ref.getPSF_Image(501, x=2048.5, y=2048.5).array

    FITS_PSF_SCI = output_dir + '/%s.centPSF.fits' %sciname
    fits.HDUList([fits.PrimaryHDU(data=psf_image_sci, header=None)]).writeto(FITS_PSF_SCI, overwrite=True)

    # rotate PSF model to align resampled image
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
    _phdr = fits.getheader(FITS_oSCI, ext=0)
    _w = Read_WCS.RW(_phdr, VERBOSE_LEVEL=1)
    x0, y0 = 0.5 + int(_phdr['NAXIS1'])/2.0, 0.5 + int(_phdr['NAXIS2'])/2.0
    ra0, dec0 = _w.all_pix2world(np.array([[x0, y0]]), 1)[0]
    skyN_vector = calculate_skyN_vector(wcshdr=_phdr, x_start=x0, y_start=y0)

    _phdr = fits.getheader(FITS_SCI, ext=0)
    _w = Read_WCS.RW(_phdr, VERBOSE_LEVEL=1)
    x1, y1 = _w.all_world2pix(np.array([[ra0, dec0]]), 1)[0]
    skyN_vectorp = calculate_skyN_vector(wcshdr=_phdr, x_start=x1, y_start=y1)
    PATTERN_ROTATE_ANGLE = calculate_rotate_angle(vector_ref=skyN_vector, vector_obj=skyN_vectorp)

    # perform rotation to get rotated psf model for resampled science
    FITS_PSF_SCI = output_dir + '/%s.centPSF.fits' %sciname
    PixA_PSF_SCI = fits.getdata(FITS_PSF_SCI, ext=0).T
    PSF_PSF_ReSCI = Image_ZoomRotate.IZR(PixA_obj=PixA_PSF_SCI, ZOOM_SCALE_X=1., \
        ZOOM_SCALE_Y=1., PATTERN_ROTATE_ANGLE=PATTERN_ROTATE_ANGLE, \
        RESAMPLING_TYPE='BILINEAR', FILL_VALUE=0.0, VERBOSE_LEVEL=1)[0]

    FITS_PSF_ReSCI = output_dir + '/%s.resamp.centPSF.fits' %sciname
    fits.HDUList([fits.PrimaryHDU(data=PSF_PSF_ReSCI.T, header=None)]).writeto(FITS_PSF_ReSCI, overwrite=True)

    # *** step1b. cross convolution
    FITS_lREF = input_dir + '/%s.sciE.skysub.fits' % refname
    FITS_lSCI = output_dir + '/%s.sciE.skysub.resamp.fits' % sciname

    FITS_PSF_lREF = output_dir + '/%s.centPSF.fits' % refname
    FITS_PSF_lSCI = output_dir + '/%s.resamp.centPSF.fits' % sciname

    FITS_lREF_convd = output_dir + '/%s.sciE.skysub.crossConvd.fits' % refname
    FITS_lSCI_convd = output_dir + '/%s.sciE.skysub.resamp.crossConvd.fits' % sciname

    PixA_lREF = fits.getdata(FITS_lREF, ext=0).T
    PixA_lSCI = fits.getdata(FITS_lSCI, ext=0).T

    PixA_PSF_lREF = fits.getdata(FITS_PSF_lREF, ext=0).T
    PixA_PSF_lSCI = fits.getdata(FITS_PSF_lSCI, ext=0).T

    # convolve (resampled) science psf on reference    
    PixA_lREF_convd = cp.asnumpy(convolve_fft_cuda(
        PixA_Inp_GPU=cp.array(PixA_lREF, dtype=cp.float64), 
        KERNEL_GPU=cp.array(PixA_PSF_lSCI, dtype=cp.float64),
        PAD_FILL_VALUE=0., NORMALIZE_KERNEL=True, FFT_BACKEND="cupy"))

    if FITS_lREF_convd is not None:
        with fits.open(FITS_lREF) as hdl:
            _message = 'Convolving image ... \n # %s' %FITS_lREF_convd
            print('\nMeLOn CheckPoint: %s!' %_message)
            hdl[0].data[:, :] = PixA_lREF_convd.T
            hdl.writeto(FITS_lREF_convd, overwrite=True)
        
    # convolve reference psf on (resampled) science
    PixA_lSCI_convd = cp.asnumpy(convolve_fft_cuda(
        PixA_Inp_GPU=cp.array(PixA_lSCI, dtype=cp.float64), 
        KERNEL_GPU=cp.array(PixA_PSF_lREF, dtype=cp.float64),
        PAD_FILL_VALUE=0., NORMALIZE_KERNEL=True, FFT_BACKEND="cupy"))

    if FITS_lSCI_convd is not None:
        with fits.open(FITS_lSCI) as hdl:
            _message = 'Convolving image ... \n # %s' %FITS_lSCI_convd
            print('\nMeLOn CheckPoint: %s!' %_message)
            hdl[0].data[:, :] = PixA_lSCI_convd.T
            hdl.writeto(FITS_lSCI_convd, overwrite=True)

    # *** step2. image subtraction using sfft
    FITS_lR = input_dir + '/%s.sciE.skysub.fits' % refname                    # use full
    FITS_lS = output_dir + '/%s.sciE.skysub.resamp.fits' % sciname            # use full

    FITS_dR = input_dir + '/%s.sciE.skysub.detmask.fits'  %refname            # use full
    FITS_dS = output_dir + '/%s.sciE.skysub.detmask.resamp.fits'  %sciname    # use full

    FITS_R = output_dir + '/%s.sciE.skysub.crossConvd.fits'  %refname         # use full
    FITS_S = output_dir + '/%s.sciE.skysub.resamp.crossConvd.fits'  %sciname  # use full

    PixA_dR = fits.getdata(FITS_dR, ext=0).T
    PixA_dS = fits.getdata(FITS_dS, ext=0).T

    LYMASK_BKG = np.logical_or(PixA_dR == 0, PixA_dS < 0.1)   # background-mask
    LYMASK_RDET = ~LYMASK_BKG

    FITS_DIFF = FITS_S[:-5] + '.polysfftdiff.fits'
    FITS_Solution = FITS_S[:-5] + '.polysfftsolution.fits'
    FITS_DCDIFF = FITS_S[:-5] + '.polysfftdiff.DeCorrelated.fits'

    # zero-out background to create masked image pair 
    TDIR = mkdtemp(suffix=None, prefix='mask', dir=None)
    FITS_mREF = TDIR + '%s.masked.fits' %(pa.basename(FITS_R)[:-5])
    FITS_mSCI = TDIR + '%s.masked.fits' %(pa.basename(FITS_S)[:-5])

    with fits.open(FITS_R) as hdl:
        _PixA = hdl[0].data.T
        _PixA[LYMASK_BKG] = 0.0
        hdl[0].data[:, :] = _PixA.T
        hdl.writeto(FITS_mREF, overwrite=True)

    with fits.open(FITS_S) as hdl:
        _PixA = hdl[0].data.T
        _PixA[LYMASK_BKG] = 0.0
        hdl[0].data[:, :] = _PixA.T
        hdl.writeto(FITS_mSCI, overwrite=True)

    # configuration for sfft subtraction
    ForceConv = 'REF'       # convolve which side, 'SCI' or 'REF'
    GKerHW = 9
    KerPolyOrder = 2        # polynomial degree for matching kerenl spatial variation 
    BGPolyOrder = 0         # polynomial degree for differential background variation, trivial here
    ConstPhotRatio = True   # constant flux scaling?

    BACKEND_4SUBTRACT = 'Cupy'      # FIXME: Please use 'Numpy' if no gpu device avaiable
    CUDA_DEVICE_4SUBTRACT = '0'     # gpu device index, only for Cupy backend
    NUM_CPU_THREADS_4SUBTRACT = 8   # number of cpu threads, only for Numpy backend

    # run polynomial form sfft subtraction 
    Customized_Packet.CP(FITS_REF=FITS_R, FITS_SCI=FITS_S, FITS_mREF=FITS_mREF, FITS_mSCI=FITS_mSCI, \
        ForceConv=ForceConv, GKerHW=GKerHW, FITS_DIFF=FITS_DIFF, FITS_Solution=FITS_Solution, \
        KerPolyOrder=KerPolyOrder, BGPolyOrder=BGPolyOrder, ConstPhotRatio=ConstPhotRatio, \
        BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, CUDA_DEVICE_4SUBTRACT=CUDA_DEVICE_4SUBTRACT, \
        NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT)
    os.system('rm -rf %s' %TDIR)

    # *** step3. noise decorrelation & SNR estimation 
    # run noise decorrelation for polynomial form sfft subtraction 
    PixA_PSF_lREF = fits.getdata(FITS_PSF_lREF, ext=0).T
    PixA_PSF_lSCI = fits.getdata(FITS_PSF_lSCI, ext=0).T

    PixA_lR = fits.getdata(FITS_lR, ext=0).T # use stamp
    PixA_lS = fits.getdata(FITS_lS, ext=0).T # use stamp
    PixA_DIFF = fits.getdata(FITS_DIFF, ext=0).T 

    bkgsig_REF = SkyLevel_Estimator.SLE(PixA_obj=PixA_lR)[1]
    bkgsig_SCI = SkyLevel_Estimator.SLE(PixA_obj=PixA_lS)[1]

    N0, N1 = PixA_lS.shape
    XY_q = np.array([[N0/2.+0.5, N1/2.+0.5]])
    MKerStack = Realize_MatchingKernel(XY_q).FromFITS(FITS_Solution=FITS_Solution)
    MK_Fin = MKerStack[0]

    DCKer = DeCorrelation_Calculator.DCC(MK_JLst=[PixA_PSF_lREF], SkySig_JLst=[bkgsig_SCI], \
        MK_ILst=[PixA_PSF_lSCI], SkySig_ILst=[bkgsig_REF], MK_Fin=MK_Fin, \
        KERatio=2.0, VERBOSE_LEVEL=2)

    PixA_DCDIFF = cp.asnumpy(convolve_fft_cuda(
        PixA_Inp_GPU=cp.array(PixA_DIFF, dtype=cp.float64), 
        KERNEL_GPU=cp.array(DCKer, dtype=cp.float64),
        PAD_FILL_VALUE=0., NORMALIZE_KERNEL=True, FFT_BACKEND="cupy"))

    with fits.open(FITS_DIFF) as hdl:
        hdl[0].data[:, :] = PixA_DCDIFF.T
        hdl.writeto(FITS_DCDIFF, overwrite=True)

    # *** step4. noise decorrelation & SNR estimation 
    # roughly estimate the SNR map for the decorrelated difference image
    # WARNING: the noise propagation is highly simplified.

    GAIN = 1.0  
    PixA_varREF = np.clip(PixA_lR/GAIN, a_min=0.0, a_max=None) + bkgsig_REF**2
    PixA_varSCI = np.clip(PixA_lS/GAIN, a_min=0.0, a_max=None) + bkgsig_SCI**2
    PixA_DIFF_Noise = np.sqrt(PixA_varREF + PixA_varSCI)

    FITS_DSNR = FITS_DIFF[:-5] + '.DeCorrelated.SNR.fits'
    PixA_DSNR = PixA_DCDIFF / PixA_DIFF_Noise
    with fits.open(FITS_DIFF) as hdl:
        hdl[0].data[:, :] = PixA_DSNR.T
        hdl.writeto(FITS_DSNR, overwrite=True)

    return None

def WorkFlow_NVTX(refname, sciname, input_dir, output_dir, aux_dir, config_file):

    # *** step0a. prepare mapping for resampling [science image & detection mask onto reference frame]
    with nvtx.annotate("Step0a_resamp_mapping)", color="#FFD900"):
        FITS_REF = input_dir + "/%s.sciE.skysub.fits" % refname
        FITS_oSCI = input_dir + "/%s.sciE.skysub.fits" % sciname

        FITS_REF_DMASK = input_dir + "/%s.sciE.skysub.detmask.fits" % refname
        FITS_oSCI_DMASK = input_dir + "/%s.sciE.skysub.detmask.fits" % sciname

        assert pa.exists(FITS_REF) and pa.exists(FITS_oSCI) 
        assert pa.exists(FITS_REF_DMASK) and pa.exists(FITS_oSCI_DMASK)

        FITS_SCI = output_dir + "/%s.sciE.skysub.resamp.fits" % sciname
        FITS_SCI_DMASK = output_dir + "/%s.sciE.skysub.detmask.resamp.fits" % sciname

        # create the mapping for object to target 
        CR1 = Cuda_Resampling(FITS_obj=FITS_oSCI, FITS_targ=FITS_REF, METHOD='BILINEAR', VERBOSE_LEVEL=1)
        CR2 = Cuda_Resampling(FITS_obj=FITS_oSCI_DMASK, FITS_targ=FITS_REF_DMASK, METHOD='BILINEAR', VERBOSE_LEVEL=1)
        
        # todo: the two mappings are same
        PixA_Eobj1, MappingDICT1 = CR1.mapping()
        PixA_Eobj2, MappingDICT2 = CR2.mapping()

    # *** step0b. resampling [science image & detection mask onto reference frame]
    with nvtx.annotate("Step0b_resamp_main", color="#8337EC"):
        PixA_resamp1 = CR1.resampling(PixA_Eobj=PixA_Eobj1, MappingDICT=MappingDICT1)
        PixA_resamp2 = CR2.resampling(PixA_Eobj=PixA_Eobj2, MappingDICT=MappingDICT2)

    with fits.open(FITS_REF) as hdl:
        PixA_resamp1[PixA_resamp1 == 0.] = np.nan
        hdl[0].data[:, :] = PixA_resamp1.T
        hdl.writeto(FITS_SCI, overwrite=True)
    
    with fits.open(FITS_REF_DMASK) as hdl:
        hdl[0].data[:, :] = PixA_resamp2.T
        hdl.writeto(FITS_SCI_DMASK, overwrite=True)

    # *** step1a. extract psf models
    with nvtx.annotate("Step1a_croosConv_extractPSF", color="#1D90FF"):
        # retrieve psf model at reference image center
        util_ref = roman_utils(config_file=config_file, image_name='%s.fits' %refname)
        psf_image_ref = util_ref.getPSF_Image(501, x=2048.5, y=2048.5).array

        FITS_PSF_REF = output_dir + '/%s.centPSF.fits' %refname
        fits.HDUList([fits.PrimaryHDU(data=psf_image_ref, header=None)]).writeto(FITS_PSF_REF, overwrite=True)

        # retrieve psf model at (unresampled) science image center
        util_sci = roman_utils(config_file=config_file, image_name='%s.fits' %sciname)
        psf_image_sci = util_ref.getPSF_Image(501, x=2048.5, y=2048.5).array

        FITS_PSF_SCI = output_dir + '/%s.centPSF.fits' %sciname
        fits.HDUList([fits.PrimaryHDU(data=psf_image_sci, header=None)]).writeto(FITS_PSF_SCI, overwrite=True)

        # rotate PSF model to align resampled image
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
        _phdr = fits.getheader(FITS_oSCI, ext=0)
        _w = Read_WCS.RW(_phdr, VERBOSE_LEVEL=1)
        x0, y0 = 0.5 + int(_phdr['NAXIS1'])/2.0, 0.5 + int(_phdr['NAXIS2'])/2.0
        ra0, dec0 = _w.all_pix2world(np.array([[x0, y0]]), 1)[0]
        skyN_vector = calculate_skyN_vector(wcshdr=_phdr, x_start=x0, y_start=y0)

        _phdr = fits.getheader(FITS_SCI, ext=0)
        _w = Read_WCS.RW(_phdr, VERBOSE_LEVEL=1)
        x1, y1 = _w.all_world2pix(np.array([[ra0, dec0]]), 1)[0]
        skyN_vectorp = calculate_skyN_vector(wcshdr=_phdr, x_start=x1, y_start=y1)
        PATTERN_ROTATE_ANGLE = calculate_rotate_angle(vector_ref=skyN_vector, vector_obj=skyN_vectorp)

        # perform rotation to get rotated psf model for resampled science
        FITS_PSF_SCI = output_dir + '/%s.centPSF.fits' %sciname
        PixA_PSF_SCI = fits.getdata(FITS_PSF_SCI, ext=0).T
        PSF_PSF_ReSCI = Image_ZoomRotate.IZR(PixA_obj=PixA_PSF_SCI, ZOOM_SCALE_X=1., \
            ZOOM_SCALE_Y=1., PATTERN_ROTATE_ANGLE=PATTERN_ROTATE_ANGLE, \
            RESAMPLING_TYPE='BILINEAR', FILL_VALUE=0.0, VERBOSE_LEVEL=1)[0]

        FITS_PSF_ReSCI = output_dir + '/%s.resamp.centPSF.fits' %sciname
        fits.HDUList([fits.PrimaryHDU(data=PSF_PSF_ReSCI.T, header=None)]).writeto(FITS_PSF_ReSCI, overwrite=True)

    # *** step1b. cross convolution
    with nvtx.annotate("Step1b_croosConv_main", color="#C51B8A"):
        FITS_lREF = input_dir + '/%s.sciE.skysub.fits' % refname
        FITS_lSCI = output_dir + '/%s.sciE.skysub.resamp.fits' % sciname

        FITS_PSF_lREF = output_dir + '/%s.centPSF.fits' % refname
        FITS_PSF_lSCI = output_dir + '/%s.resamp.centPSF.fits' % sciname

        FITS_lREF_convd = output_dir + '/%s.sciE.skysub.crossConvd.fits' % refname
        FITS_lSCI_convd = output_dir + '/%s.sciE.skysub.resamp.crossConvd.fits' % sciname

        PixA_lREF = fits.getdata(FITS_lREF, ext=0).T
        PixA_lSCI = fits.getdata(FITS_lSCI, ext=0).T

        PixA_PSF_lREF = fits.getdata(FITS_PSF_lREF, ext=0).T
        PixA_PSF_lSCI = fits.getdata(FITS_PSF_lSCI, ext=0).T

        # convolve (resampled) science psf on reference
        PixA_lREF_convd = cp.asnumpy(convolve_fft_cuda(
            PixA_Inp_GPU=cp.array(PixA_lREF, dtype=cp.float64), 
            KERNEL_GPU=cp.array(PixA_PSF_lSCI, dtype=cp.float64),
            PAD_FILL_VALUE=0., NORMALIZE_KERNEL=True, FFT_BACKEND="cupy"))

        if FITS_lREF_convd is not None:
            with fits.open(FITS_lREF) as hdl:
                _message = 'Convolving image ... \n # %s' %FITS_lREF_convd
                print('\nMeLOn CheckPoint: %s!' %_message)
                hdl[0].data[:, :] = PixA_lREF_convd.T
                hdl.writeto(FITS_lREF_convd, overwrite=True)
            
        # convolve reference psf on (resampled) science
        PixA_lSCI_convd = cp.asnumpy(convolve_fft_cuda(
            PixA_Inp_GPU=cp.array(PixA_lSCI, dtype=cp.float64), 
            KERNEL_GPU=cp.array(PixA_PSF_lREF, dtype=cp.float64),
            PAD_FILL_VALUE=0., NORMALIZE_KERNEL=True, FFT_BACKEND="cupy"))

        if FITS_lSCI_convd is not None:
            with fits.open(FITS_lSCI) as hdl:
                _message = 'Convolving image ... \n # %s' %FITS_lSCI_convd
                print('\nMeLOn CheckPoint: %s!' %_message)
                hdl[0].data[:, :] = PixA_lSCI_convd.T
                hdl.writeto(FITS_lSCI_convd, overwrite=True)

    # *** step2b. image subtraction using sfft
    with nvtx.annotate("Step2_sfft_main", color="#233ED9"):
        FITS_lR = input_dir + '/%s.sciE.skysub.fits' % refname                    # use full
        FITS_lS = output_dir + '/%s.sciE.skysub.resamp.fits' % sciname            # use full

        FITS_dR = input_dir + '/%s.sciE.skysub.detmask.fits'  %refname            # use full
        FITS_dS = output_dir + '/%s.sciE.skysub.detmask.resamp.fits'  %sciname    # use full

        FITS_R = output_dir + '/%s.sciE.skysub.crossConvd.fits'  %refname         # use full
        FITS_S = output_dir + '/%s.sciE.skysub.resamp.crossConvd.fits'  %sciname  # use full

        PixA_dR = fits.getdata(FITS_dR, ext=0).T
        PixA_dS = fits.getdata(FITS_dS, ext=0).T

        LYMASK_BKG = np.logical_or(PixA_dR == 0, PixA_dS < 0.1)   # background-mask
        LYMASK_RDET = ~LYMASK_BKG

        FITS_DIFF = FITS_S[:-5] + '.polysfftdiff.fits'
        FITS_Solution = FITS_S[:-5] + '.polysfftsolution.fits'
        FITS_DCDIFF = FITS_S[:-5] + '.polysfftdiff.DeCorrelated.fits'

        # zero-out background to create masked image pair 
        TDIR = mkdtemp(suffix=None, prefix='mask', dir=None)
        FITS_mREF = TDIR + '%s.masked.fits' %(pa.basename(FITS_R)[:-5])
        FITS_mSCI = TDIR + '%s.masked.fits' %(pa.basename(FITS_S)[:-5])

        with fits.open(FITS_R) as hdl:
            _PixA = hdl[0].data.T
            _PixA[LYMASK_BKG] = 0.0
            hdl[0].data[:, :] = _PixA.T
            hdl.writeto(FITS_mREF, overwrite=True)

        with fits.open(FITS_S) as hdl:
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
        Customized_Packet.CP(FITS_REF=FITS_R, FITS_SCI=FITS_S, FITS_mREF=FITS_mREF, FITS_mSCI=FITS_mSCI, \
            ForceConv=ForceConv, GKerHW=GKerHW, FITS_DIFF=FITS_DIFF, FITS_Solution=FITS_Solution, \
            KerPolyOrder=KerPolyOrder, BGPolyOrder=BGPolyOrder, ConstPhotRatio=ConstPhotRatio, \
            BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, CUDA_DEVICE_4SUBTRACT=CUDA_DEVICE_4SUBTRACT, \
            NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT)
        os.system('rm -rf %s' %TDIR)

    # *** step3. noise decorrelation & SNR estimation 
    with nvtx.annotate("Step3_decorr", color="#65BD63"):
        # run noise decorrelation for polynomial form sfft subtraction 
        PixA_PSF_lREF = fits.getdata(FITS_PSF_lREF, ext=0).T
        PixA_PSF_lSCI = fits.getdata(FITS_PSF_lSCI, ext=0).T

        PixA_lR = fits.getdata(FITS_lR, ext=0).T # use stamp
        PixA_lS = fits.getdata(FITS_lS, ext=0).T # use stamp
        PixA_DIFF = fits.getdata(FITS_DIFF, ext=0).T 

        bkgsig_REF = SkyLevel_Estimator.SLE(PixA_obj=PixA_lR)[1]
        bkgsig_SCI = SkyLevel_Estimator.SLE(PixA_obj=PixA_lS)[1]

        N0, N1 = PixA_lS.shape
        XY_q = np.array([[N0/2.+0.5, N1/2.+0.5]])
        MKerStack = Realize_MatchingKernel(XY_q).FromFITS(FITS_Solution=FITS_Solution)
        MK_Fin = MKerStack[0]

        DCKer = DeCorrelation_Calculator.DCC(MK_JLst=[PixA_PSF_lREF], SkySig_JLst=[bkgsig_SCI], \
            MK_ILst=[PixA_PSF_lSCI], SkySig_ILst=[bkgsig_REF], MK_Fin=MK_Fin, \
            KERatio=2.0, VERBOSE_LEVEL=2)

        PixA_DCDIFF = cp.asnumpy(convolve_fft_cuda(
            PixA_Inp_GPU=cp.array(PixA_DIFF, dtype=cp.float64), 
            KERNEL_GPU=cp.array(DCKer, dtype=cp.float64),
            PAD_FILL_VALUE=0., NORMALIZE_KERNEL=True, FFT_BACKEND="cupy"))

        with fits.open(FITS_DIFF) as hdl:
            hdl[0].data[:, :] = PixA_DCDIFF.T
            hdl.writeto(FITS_DCDIFF, overwrite=True)

    # *** step4. noise decorrelation & SNR estimation 
    with nvtx.annotate("Step4_snr", color="#00D0FF"):
        # roughly estimate the SNR map for the decorrelated difference image
        # WARNING: the noise propagation is highly simplified.

        GAIN = 1.0  
        PixA_varREF = np.clip(PixA_lR/GAIN, a_min=0.0, a_max=None) + bkgsig_REF**2
        PixA_varSCI = np.clip(PixA_lS/GAIN, a_min=0.0, a_max=None) + bkgsig_SCI**2
        PixA_DIFF_Noise = np.sqrt(PixA_varREF + PixA_varSCI)

        FITS_DSNR = FITS_DIFF[:-5] + '.DeCorrelated.SNR.fits'
        PixA_DSNR = PixA_DCDIFF / PixA_DIFF_Noise
        with fits.open(FITS_DIFF) as hdl:
            hdl[0].data[:, :] = PixA_DSNR.T
            hdl.writeto(FITS_DSNR, overwrite=True)

    return None

# WorkFlow(refname=refname, sciname=sciname, input_dir=input_dir, 
#     output_dir=output_dir, aux_dir=aux_dir, config_file=config_file)

print('************ Start NVTX! ************')
WorkFlow_NVTX(refname=refname, sciname=sciname, input_dir=input_dir, 
    output_dir=output_dir, aux_dir=aux_dir, config_file=config_file)
print('************ End NVTX! ************')

"""
# command to profile the workflow using Nvidia Nsight Systems
nsys profile --trace-fork-before-exec=true --python-backtrace=cuda --python-sampling=true --trace=cuda,nvtx,cublas,cusparse,cudnn,cudla,cusolver,opengl,openacc,openmp,osrt,mpi,nvvideo,vulkan,python-gil python sfftPureSub4Roman.py

"""
