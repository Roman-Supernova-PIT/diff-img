import time
import warnings
import cupy as cp
import numpy as np
from math import floor
from astropy.wcs import WCS, FITSFixedWarning
from CupyWCSTransform import Cupy_WCS_Transform

__last_update__ = "2024-09-22"
__author__ = "Lei Hu <leihu@andrew.cmu.edu>"

class Cuda_Resampling:
    def __init__(self, RESAMP_METHOD="BILINEAR", VERBOSE_LEVEL=2):
        """
        Image resampling using CUDA.
        """
        self.RESAMP_METHOD = RESAMP_METHOD
        self.VERBOSE_LEVEL = VERBOSE_LEVEL
        return None
    
    def projection_cd(self, hdr_obj, hdr_targ, CDKEY="CD"):
        """Mapping the target pixel centers to the object frame using Cupy for CD WCS (NO distortion)"""
        NTX = int(hdr_targ["NAXIS1"]) 
        NTY = int(hdr_targ["NAXIS2"])

        XX_targ_GPU, YY_targ_GPU = cp.meshgrid(
            cp.arange(0, NTX) + 1., 
            cp.arange(0, NTY) + 1., 
            indexing='ij'
        )
        
        CWT = Cupy_WCS_Transform()
        # * perform CD transformation through target WCS
        if True:
            # read header [target]
            KEYDICT, CD_GPU = CWT.read_cd_wcs(hdr_wcs=hdr_targ, CDKEY=CDKEY)
            CRPIX1_targ, CRPIX2_targ = KEYDICT["CRPIX1"], KEYDICT["CRPIX2"]
            
            # relative to reference point [target]
            u_GPU, v_GPU = XX_targ_GPU.flatten() - CRPIX1_targ, YY_targ_GPU.flatten() - CRPIX2_targ
            
            # CD transformation [target]
            x_GPU, y_GPU = CWT.cd_transform(IMAGE_X_GPU=u_GPU, IMAGE_Y_GPU=v_GPU, CD_GPU=CD_GPU)

        # * perform CD^-1 transformation through object WCS
        if True:
            # read header [object]
            KEYDICT, CD_GPU = CWT.read_cd_wcs(hdr_wcs=hdr_obj, CDKEY=CDKEY)
            CRPIX1_obj, CRPIX2_obj = KEYDICT["CRPIX1"], KEYDICT["CRPIX2"]

            # relative to reference point, consider offset between the WCS reference points [object]
            # WARNING: the offset calculations may be wrong when the WCS reference points are very close to the N/S Poles!
            x_GPU += (CRPIX1_targ - CRPIX1_obj + 180.0) % 360.0 - 180.0   # from using target reference point to object
            y_GPU += CRPIX2_targ - CRPIX2_obj                             # ~

            # inverse CD transformation [object]
            u_GPU, v_GPU = CWT.cd_transform_inv(WORLD_X_GPU=x_GPU, WORLD_Y_GPU=y_GPU, CD_GPU=CD_GPU)
            
            # relative to image origin [object]
            XX_proj_GPU = CRPIX1_obj + u_GPU.reshape((NTX, NTY))
            YY_proj_GPU = CRPIX2_obj + v_GPU.reshape((NTX, NTY))
        
        return XX_proj_GPU, YY_proj_GPU

    def projection_sip(self, hdr_obj, hdr_targ, Nsamp=1024, RANDOM_SEED=10086):
        """Mapping the target pixel centers to the object frame using Cupy for SIP WCS"""
        NTX = int(hdr_targ["NAXIS1"]) 
        NTY = int(hdr_targ["NAXIS2"])

        XX_targ_GPU, YY_targ_GPU = cp.meshgrid(
            cp.arange(0, NTX) + 1., 
            cp.arange(0, NTY) + 1., 
            indexing='ij'
        )
        
        CWT = Cupy_WCS_Transform()
        # * perform forward transformation (+CD) through target WCS
        if True:
            # read header [target]
            KEYDICT, CD_GPU, A_SIP_GPU, B_SIP_GPU = CWT.read_sip_wcs(hdr_wcs=hdr_targ)
            CRPIX1_targ, CRPIX2_targ = KEYDICT["CRPIX1"], KEYDICT["CRPIX2"]
            CRVAL1_targ, CRVAL2_targ = KEYDICT["CRVAL1"], KEYDICT["CRVAL2"]
            
            # relative to reference point [target]
            u_GPU, v_GPU = XX_targ_GPU.flatten() - CRPIX1_targ, YY_targ_GPU.flatten() - CRPIX2_targ
            
            # forward transformation for target grid of pixel centers [target]
            U_GPU, V_GPU = CWT.sip_forward_transform(u_GPU=u_GPU, v_GPU=v_GPU, A_SIP_GPU=A_SIP_GPU, B_SIP_GPU=B_SIP_GPU)
            
            # plus CD transformation [target]
            x_GPU, y_GPU = CWT.cd_transform(IMAGE_X_GPU=U_GPU, IMAGE_Y_GPU=V_GPU, CD_GPU=CD_GPU)
        
        # * perform backward transformation (+CD^-1) through object WCS
        if True:
            # read header [object]
            KEYDICT, CD_GPU, A_SIP_GPU, B_SIP_GPU = CWT.read_sip_wcs(hdr_wcs=hdr_obj)
            CRPIX1_obj, CRPIX2_obj = KEYDICT["CRPIX1"], KEYDICT["CRPIX2"]
            CRVAL1_obj, CRVAL2_obj = KEYDICT["CRVAL1"], KEYDICT["CRVAL2"]

            # sampling random coordinates [object]
            u_GPU, v_GPU = CWT.random_coord_sampling(N0=KEYDICT["N0"], N1=KEYDICT["N1"], 
                CRPIX1=CRPIX1_obj, CRPIX2=CRPIX2_obj, Nsamp=Nsamp, RANDOM_SEED=RANDOM_SEED)
            
            # fit polynomial form backward transformation [object]
            U_GPU, V_GPU = CWT.sip_forward_transform(u_GPU=u_GPU, v_GPU=v_GPU, A_SIP_GPU=A_SIP_GPU, B_SIP_GPU=B_SIP_GPU)
            AP_lstsq_GPU, BP_lstsq_GPU = CWT.lstsq_sip_backward_transform(u_GPU=u_GPU, v_GPU=v_GPU, 
                U_GPU=U_GPU, V_GPU=V_GPU, A_ORDER=KEYDICT["A_ORDER"], B_ORDER=KEYDICT["B_ORDER"])[2:4]

            # relative to reference point, consider offset between the WCS reference points [object]
            # WARNING: the offset calculations may be wrong when the WCS reference points are very close to the N/S Poles!
            # TODO: this is not accruate, ~ 1pix.
            offset1 = ((CRVAL1_targ - CRVAL1_obj + 180.0) % 360.0 - 180.0) * cp.cos(cp.deg2rad((CRVAL2_targ + CRVAL2_obj)/2.))
            offset2 = CRVAL2_targ - CRVAL2_obj
            x_GPU += offset1
            y_GPU += offset2

            # inverse CD transformation [object]
            U_GPU, V_GPU = CWT.cd_transform_inv(WORLD_X_GPU=x_GPU, WORLD_Y_GPU=y_GPU, CD_GPU=CD_GPU)
            
            # backward transformation [object]
            FP_UV_GPU = CWT.sip_backward_matrix(U_GPU=U_GPU, V_GPU=V_GPU, ORDER=KEYDICT["A_ORDER"])
            GP_UV_GPU = CWT.sip_backward_matrix(U_GPU=U_GPU, V_GPU=V_GPU, ORDER=KEYDICT["B_ORDER"])

            u_GPU, v_GPU = CWT.sip_backward_transform(U_GPU=U_GPU, V_GPU=V_GPU, 
                FP_UV_GPU=FP_UV_GPU, GP_UV_GPU=GP_UV_GPU, AP_GPU=AP_lstsq_GPU, BP_GPU=BP_lstsq_GPU)
            
            # relative to image origin [object]
            XX_proj_GPU = CRPIX1_obj + u_GPU.reshape((NTX, NTY))
            YY_proj_GPU = CRPIX2_obj + v_GPU.reshape((NTX, NTY))

        return XX_proj_GPU, YY_proj_GPU

    def projection_astropy(self, hdr_obj, hdr_targ):
        """Mapping the target pixel centers to the object frame using Astropy"""
        # * read object WCS and target WCS
        def _readWCS(hdr, VERBOSE_LEVEL=2):
            with warnings.catch_warnings():
                if VERBOSE_LEVEL in [0, 1]: behavior = 'ignore'
                if VERBOSE_LEVEL in [2]: behavior = 'default'
                warnings.filterwarnings(behavior, category=FITSFixedWarning)

                if hdr['CTYPE1'] == 'RA---TAN' and 'PV1_0' in hdr:
                    _hdr = hdr.copy()
                    _hdr['CTYPE1'] = 'RA---TPV'
                    _hdr['CTYPE2'] = 'DEC--TPV'
                else: _hdr = hdr
                w = WCS(_hdr)
            return w

        w_obj = _readWCS(hdr=hdr_obj, VERBOSE_LEVEL=self.VERBOSE_LEVEL)
        w_targ = _readWCS(hdr=hdr_targ, VERBOSE_LEVEL=self.VERBOSE_LEVEL)

        # * maaping target pixel centers to the object frame
        NTX = int(hdr_targ["NAXIS1"]) 
        NTY = int(hdr_targ["NAXIS2"])
        
        XX_targ, YY_targ = np.meshgrid(np.arange(0, NTX)+1., np.arange(0, NTY)+1., indexing='ij')
        XY_targ = np.array([XX_targ.flatten(), YY_targ.flatten()]).T
        XY_proj = w_obj.all_world2pix(w_targ.all_pix2world(XY_targ, 1), 1)

        XX_proj = XY_proj[:, 0].reshape((NTX, NTY))
        YY_proj = XY_proj[:, 1].reshape((NTX, NTY))

        XX_proj_GPU = cp.array(XX_proj, dtype=np.float64)
        YY_proj_GPU = cp.array(YY_proj, dtype=np.float64)

        return XX_proj_GPU, YY_proj_GPU

    def frame_extension(self, XX_proj_GPU, YY_proj_GPU, PixA_obj_GPU):
        """Extend the object frame for resampling"""
        NTX, NTY = XX_proj_GPU.shape
        NOX, NOY = PixA_obj_GPU.shape

        # * padding the object frame
        if self.RESAMP_METHOD == 'BILINEAR':
            KERHW = (1, 1)

        if self.RESAMP_METHOD == 'LANCZOS3':
            KERHW = (3, 3)

        # find the root index and shift with maximal kernel halfwidth (KERHW)
        # Note: the index ranges (RMIN --- RMAX) and (CMIN --- CMAX) can cover
        #       all pixels that interpolation may be use.

        RMIN = (floor(XX_proj_GPU.min().item()) - 1) - KERHW[0]
        RMAX = (floor(XX_proj_GPU.max().item()) - 1) + KERHW[0]
        RPAD = (-np.min([RMIN, 0]), np.max([RMAX - (NOX - 1), 0]))

        CMIN = (floor(YY_proj_GPU.min().item()) - 1) - KERHW[1]
        CMAX = (floor(YY_proj_GPU.max().item()) - 1) + KERHW[1]
        CPAD = (-np.min([CMIN, 0]), np.max([CMAX - (NOY - 1), 0]))

        PAD_WIDTH = (RPAD, CPAD)    
        PixA_Eobj_GPU = cp.pad(PixA_obj_GPU, PAD_WIDTH, mode='constant', constant_values=0.)
        NEOX, NEOY = PixA_Eobj_GPU.shape

        XX_Eproj_GPU = XX_proj_GPU + PAD_WIDTH[0][0]
        YY_Eproj_GPU = YY_proj_GPU + PAD_WIDTH[1][0]

        RMIN_E = (floor(XX_Eproj_GPU.min().item()) - 1) - KERHW[0]
        RMAX_E = (floor(XX_Eproj_GPU.max().item()) - 1) + KERHW[0]

        CMIN_E = (floor(YY_Eproj_GPU.min().item()) - 1) - KERHW[1]
        CMAX_E = (floor(YY_Eproj_GPU.max().item()) - 1) + KERHW[1]

        assert RMIN_E >= 0 and CMIN_E >= 0 
        assert RMAX_E < NEOX and CMAX_E < NEOY

        EProjDict = {}
        EProjDict['NTX'] = NTX
        EProjDict['NTY'] = NTY

        EProjDict['NOX'] = NOX
        EProjDict['NOY'] = NOY

        EProjDict['NEOX'] = NEOX
        EProjDict['NEOY'] = NEOY

        EProjDict['XX_Eproj_GPU'] = XX_Eproj_GPU
        EProjDict['YY_Eproj_GPU'] = YY_Eproj_GPU

        return PixA_Eobj_GPU, EProjDict
    
    def resampling(self, PixA_Eobj_GPU, EProjDict):
        """Resampling the object frame to the target frame using CUDA"""
        NTX = EProjDict['NTX']
        NTY = EProjDict['NTY']

        NEOX = EProjDict['NEOX']
        NEOY = EProjDict['NEOY']

        XX_Eproj_GPU = EProjDict['XX_Eproj_GPU']
        YY_Eproj_GPU = EProjDict['YY_Eproj_GPU']

        # * Cupy configuration
        MaxThreadPerB = 8
        GPUManage = lambda NT: ((NT-1)//MaxThreadPerB + 1, min(NT, MaxThreadPerB))
        BpG_PIX0, TpB_PIX0 = GPUManage(NTX)
        BpG_PIX1, TpB_PIX1 = GPUManage(NTY)
        BpG_PIX, TpB_PIX = (BpG_PIX0, BpG_PIX1), (TpB_PIX0, TpB_PIX1, 1)

        PixA_resamp_GPU = cp.zeros((NTX, NTY), dtype=np.float64)

        if self.RESAMP_METHOD == "BILINEAR":
        
            # * perform bilinear resampling using CUDA
            # input: PixA_Eobj | (NEOX, NEOY)
            # input: XX_Eproj, YY_Eproj | (NTX, NTY)
            # output: PixA_resamp | (NTX, NTY)
            
            _refdict = {'NTX': NTX, 'NTY': NTY, 'NEOX': NEOX, 'NEOY': NEOY}
            _funcstr = r"""
            extern "C" __global__ void kmain(double XX_Eproj_GPU[%(NTX)s][%(NTY)s], double YY_Eproj_GPU[%(NTX)s][%(NTY)s], 
                double PixA_Eobj_GPU[%(NEOX)s][%(NEOY)s], double PixA_resamp_GPU[%(NTX)s][%(NTY)s])
            {
                int ROW = blockIdx.x*blockDim.x+threadIdx.x;
                int COL = blockIdx.y*blockDim.y+threadIdx.y;

                int NTX = %(NTX)s;
                int NTY = %(NTY)s;
                int NEOX = %(NEOX)s;
                int NEOY = %(NEOY)s;

                if (ROW < NTX && COL < NTY) {
                
                    double x = XX_Eproj_GPU[ROW][COL];
                    double y = YY_Eproj_GPU[ROW][COL];

                    int r1 = floor(x) - 1;
                    int c1 = floor(y) - 1;
                    int r2 = r1 + 1;
                    int c2 = c1 + 1;

                    double dx = x - floor(x); 
                    double dy = y - floor(y);

                    double w11 = (1-dx) * (1-dy);
                    double w12 = (1-dx) * dy;
                    double w21 = dx * (1-dy);
                    double w22 = dx * dy;

                    PixA_resamp_GPU[ROW][COL] = w11 * PixA_Eobj_GPU[r1][c1] + w12 * PixA_Eobj_GPU[r1][c2] + 
                        w21 * PixA_Eobj_GPU[r2][c1] + w22 * PixA_Eobj_GPU[r2][c2];
                }
            }
            """
            _code = _funcstr % _refdict
            _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
            resamp_func = _module.get_function('kmain')
            
            t0 = time.time()
            resamp_func(args=(XX_Eproj_GPU, YY_Eproj_GPU, PixA_Eobj_GPU, PixA_resamp_GPU), block=TpB_PIX, grid=BpG_PIX)
            if self.VERBOSE_LEVEL in [1, 2]:
                print('MeLOn CheckPoint: Cuda resampling takes [%.6f s]' %(time.time() - t0))

        return PixA_resamp_GPU
