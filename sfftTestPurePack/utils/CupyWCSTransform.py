import cupy as cp

__last_update__ = "2024-09-22"
__author__ = "Shu Liu <shl159@pitt.edu> & Lei Hu <leihu@andrew.cmu.edu>"

class Cupy_WCS_Transform:
    def __init__(self):
        return None

    def read_cd_wcs(self, hdr_wcs, CDKEY="CD"):
        """Read CD matrix from the header"""
        assert hdr_wcs["CTYPE1"] == "RA---TAN"
        assert hdr_wcs["CTYPE2"] == "DEC--TAN"

        N0 = int(hdr_wcs["NAXIS1"])
        N1 = int(hdr_wcs["NAXIS2"])

        CRPIX1 = float(hdr_wcs["CRPIX1"])
        CRPIX2 = float(hdr_wcs["CRPIX2"])

        CRVAL1 = float(hdr_wcs["CRVAL1"])
        CRVAL2 = float(hdr_wcs["CRVAL2"])

        KEYDICT = {
            "N0": N0, "N1": N1, 
            "CRPIX1": CRPIX1, "CRPIX2": CRPIX2,
            "CRVAL1": CRVAL1, "CRVAL2": CRVAL2
        }

        CD1_1 = hdr_wcs[f"{CDKEY}1_1"]
        CD1_2 = hdr_wcs[f"{CDKEY}1_2"] if f"{CDKEY}1_2" in hdr_wcs else 0.
        CD2_1 = hdr_wcs[f"{CDKEY}2_1"] if f"{CDKEY}2_1" in hdr_wcs else 0.
        CD2_2 = hdr_wcs[f"{CDKEY}2_2"]

        CD_GPU = cp.array([
            [CD1_1, CD1_2], 
            [CD2_1, CD2_2]
        ], dtype=cp.float64)
        
        return KEYDICT, CD_GPU

    def read_sip_wcs(self, hdr_wcs):
        """Read SIP coefficients from the header"""
        assert hdr_wcs["CTYPE1"] == "RA---TAN-SIP"
        assert hdr_wcs["CTYPE2"] == "DEC--TAN-SIP"

        N0 = int(hdr_wcs["NAXIS1"])
        N1 = int(hdr_wcs["NAXIS2"])

        CRPIX1 = float(hdr_wcs["CRPIX1"])
        CRPIX2 = float(hdr_wcs["CRPIX2"])

        CRVAL1 = float(hdr_wcs["CRVAL1"])
        CRVAL2 = float(hdr_wcs["CRVAL2"])

        A_ORDER = int(hdr_wcs["A_ORDER"])
        B_ORDER = int(hdr_wcs["B_ORDER"])

        KEYDICT = {
            "N0": N0, "N1": N1, 
            "CRPIX1": CRPIX1, "CRPIX2": CRPIX2,
            "CRVAL1": CRVAL1, "CRVAL2": CRVAL2, 
            "A_ORDER": A_ORDER, "B_ORDER": B_ORDER
        }

        CD_GPU = cp.array([
            [hdr_wcs["CD1_1"], hdr_wcs["CD1_2"]], 
            [hdr_wcs["CD2_1"], hdr_wcs["CD2_2"]]
        ], dtype=cp.float64)

        A_SIP_GPU = cp.zeros((A_ORDER+1, A_ORDER+1), dtype=cp.float64)
        B_SIP_GPU = cp.zeros((B_ORDER+1, B_ORDER+1), dtype=cp.float64)

        for p in range(A_ORDER + 1):
            for q in range(0, A_ORDER - p + 1):
                keyword = f"A_{p}_{q}"
                if keyword in hdr_wcs:
                    A_SIP_GPU[p, q] = hdr_wcs[keyword]
        
        for p in range(B_ORDER + 1):
            for q in range(0, B_ORDER - p + 1):
                keyword = f"B_{p}_{q}"
                if keyword in hdr_wcs:
                    B_SIP_GPU[p, q] = hdr_wcs[keyword]
        
        return KEYDICT, CD_GPU, A_SIP_GPU, B_SIP_GPU

    def cd_transform(self, IMAGE_X_GPU, IMAGE_Y_GPU, CD_GPU):
        """CD matrix transformation from image to world"""
        WORLD_X_GPU, WORLD_Y_GPU = cp.matmul(CD_GPU, cp.array([IMAGE_X_GPU, IMAGE_Y_GPU]))
        return WORLD_X_GPU, WORLD_Y_GPU
    
    def cd_transform_inv(self, WORLD_X_GPU, WORLD_Y_GPU, CD_GPU):
        """CD matrix transformation from world to image"""
        # CD_inv_GPU = cp.linalg.inv(CD_GPU)
        CD_inv_GPU = 1. / (CD_GPU[0, 0] * CD_GPU[1, 1] - CD_GPU[0, 1] * CD_GPU[1, 0]) * \
            cp.array([[CD_GPU[1, 1], -CD_GPU[0, 1]], [-CD_GPU[1, 0], CD_GPU[0, 0]]])
        IMAGE_X_GPU, IMAGE_Y_GPU = cp.matmul(CD_inv_GPU, cp.array([WORLD_X_GPU, WORLD_Y_GPU]))
        return IMAGE_X_GPU, IMAGE_Y_GPU
    
    def sip_forward_transform(self, u_GPU, v_GPU, A_SIP_GPU, B_SIP_GPU):
        """Forward SIP transformation from (u, v) to (U, V)"""
        A_ORDER = A_SIP_GPU.shape[0] - 1
        U_GPU = u_GPU + cp.zeros_like(u_GPU)
        for p in range(A_ORDER + 1):
            for q in range(A_ORDER - p + 1):
                U_GPU += A_SIP_GPU[p, q] * u_GPU**p * v_GPU**q
            
        B_ORDER = B_SIP_GPU.shape[0] - 1
        V_GPU = v_GPU + cp.zeros_like(v_GPU)
        for p in range(B_ORDER + 1):
            for q in range(B_ORDER - p + 1):
                V_GPU += B_SIP_GPU[p, q] * u_GPU**p * v_GPU**q
        return U_GPU, V_GPU

    def sip_backward_matrix(self, U_GPU, V_GPU, ORDER):
        """Compute the backward matrix P_UV (FP_UV or GP_UV)"""
        Nsamp = len(U_GPU)
        P_UV_GPU = cp.zeros((Nsamp, (ORDER+1)*(ORDER+2)//2), dtype=cp.float64)
        idx = 0
        for p in range(ORDER + 1):
            for q in range(ORDER - p + 1):
                P_UV_GPU[:, idx] = U_GPU**p * V_GPU**q
                idx += 1
        return P_UV_GPU

    def lstsq_sip_backward_transform(self, u_GPU, v_GPU, U_GPU, V_GPU, A_ORDER, B_ORDER):
        """Compute the backward transformation from (U, V) to (u, v)"""
        FP_UV_GPU = self.sip_backward_matrix(U_GPU=U_GPU, V_GPU=V_GPU, ORDER=A_ORDER)
        GP_UV_GPU = self.sip_backward_matrix(U_GPU=U_GPU, V_GPU=V_GPU, ORDER=B_ORDER)

        AP_lstsq_GPU = cp.linalg.lstsq(FP_UV_GPU, (u_GPU - U_GPU).reshape(-1, 1), rcond=None)[0]
        BP_lstsq_GPU = cp.linalg.lstsq(GP_UV_GPU, (v_GPU - V_GPU).reshape(-1, 1), rcond=None)[0]
        return FP_UV_GPU, GP_UV_GPU, AP_lstsq_GPU, BP_lstsq_GPU

    def sip_backward_transform(self, U_GPU, V_GPU, FP_UV_GPU, GP_UV_GPU, AP_GPU, BP_GPU):
        """Backward transformation from (U, V) to (u, v)"""
        u_GPU = U_GPU + cp.matmul(FP_UV_GPU, AP_GPU)[:, 0]
        v_GPU = V_GPU + cp.matmul(GP_UV_GPU, BP_GPU)[:, 0]
        return u_GPU, v_GPU

    def random_coord_sampling(self, N0, N1, CRPIX1, CRPIX2, Nsamp=1024, RANDOM_SEED=10086):
        """Random sampling of coordinates"""
        if RANDOM_SEED is not None:
            cp.random.seed(RANDOM_SEED)
        u_GPU = cp.random.uniform(0.5, N0+0.5, Nsamp, dtype=cp.float64) - CRPIX1
        v_GPU = cp.random.uniform(0.5, N1+0.5, Nsamp, dtype=cp.float64) - CRPIX2
        return u_GPU, v_GPU
