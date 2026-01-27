# GPU_BSC.py
# GPU-accelerated BSC → ESD/EAC estimation for 60 angles, batched + vectorized
from dataclasses import dataclass
import os, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.special import spherical_jn, spherical_yn
from scipy.special import jv, yv
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# -----------------------------
# Small timing helper
# -----------------------------
class Tic:
    def __init__(self, label=""):
        self.label = label
    def __enter__(self):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        self.t0 = time.perf_counter()
    def __exit__(self, *exc):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        print(f"[TIME] {self.label}: {time.perf_counter()-self.t0:.3f}s")

# -----------------------------
# I/O helpers (int32 interleaved I/Q)
# -----------------------------
def read_iq_file_int32_interleaved(path: str) -> np.ndarray:
    A = np.fromfile(path, dtype=np.int32)
    Nz = A.size // (1104 * 192)
    IQ = A.reshape((1104, 192, Nz), order='F')  # <-- critical
    I = IQ[0::2,:,:]
    Q = IQ[1::2,:,:]
    C = I.astype(np.float32) + 1j*Q.astype(np.float32)
    return C


def write_int32_padded(path: str, vol552: np.ndarray):
    assert vol552.ndim == 3 and vol552.shape[0] <= 552 and vol552.shape[1] <= 192, \
        f"Expected [552,192,Nz]-like, got {vol552.shape}"

    H, W, Nz = vol552.shape

    outB = np.transpose(vol552, (2, 0, 1))  # [Nz,552,192]

    out_full = np.zeros((Nz, 552, 192), dtype=outB.dtype)
    out_full[:, :H, :W] = outB

    inter = np.zeros((Nz, 1104, 192), dtype=np.float32)
    inter[:, 0::2, :] = out_full.astype(np.float32, copy=False)

    inter = np.rint(inter * 1000.0)
    np.maximum(inter, 0, out=inter)
    arr_i32 = inter.astype(np.int32, copy=False)  # [Nz,1104,192]

    arr_F = np.asfortranarray(np.transpose(arr_i32, (1, 2, 0)))  # [1104,192,Nz], Fortran
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arr_F.tofile(path)


# -----------------------------
# Bessel helpers (float64 for stability)
# -----------------------------
def _x_safe_like(x: torch.Tensor):
    tiny = torch.finfo(torch.float64).tiny
    x64  = x.to(torch.float64)
    return torch.where(x64 == 0, torch.copysign(torch.tensor(tiny, dtype=torch.float64, device=x64.device), x64+0.0), x64)


def sph_bessel_j(n_max: int, x: torch.Tensor):
    xs  = _x_safe_like(x)
    xnp = xs.detach().cpu().numpy()
    out = []
    for n in range(n_max+1):
        nu = n + 0.5
        val = np.sqrt(np.pi/(2.0*xnp)) * jv(nu, xnp)
        if np.any(xnp == 0):
            val = np.where(xnp==0, 1.0 if n==0 else 0.0, val)
        out.append(val)
    out = np.stack(out, axis=0)
    return torch.from_numpy(out).to(x.device, dtype=torch.float64)


def sph_bessel_y(n_max: int, x: torch.Tensor):
    xs  = _x_safe_like(x)
    xnp = xs.detach().cpu().numpy()
    out = []
    for n in range(n_max+1):
        nu = n + 0.5
        val = np.sqrt(np.pi/(2.0*xnp)) * yv(nu, xnp)
        out.append(val)
    out = np.stack(out, axis=0)
    return torch.from_numpy(out).to(x.device, dtype=torch.float64)


def x_times_derivative_jn(j_stack: torch.Tensor):
    n_max = j_stack.shape[0]-1
    n = torch.arange(0, n_max+1, dtype=j_stack.dtype, device=j_stack.device).view(-1,1)
    j_next = torch.zeros_like(j_stack); j_next[:-1] = j_stack[1:]
    return n*j_stack - (n+1)*j_next


def x_over_j_times_jprime(j_stack: torch.Tensor) -> torch.Tensor:
    j_stack = j_stack.to(torch.float64)
    nmax = j_stack.shape[0] - 1
    dev  = j_stack.device
    n    = torch.arange(0, nmax+1, dtype=torch.float64, device=dev).view(-1,1)
    j_next = torch.zeros_like(j_stack); j_next[:-1] = j_stack[1:]
    xjprime = n*j_stack - (n+1)*j_next
    realmin = torch.finfo(torch.float64).tiny
    den = j_stack
    den_safe = torch.where(den.abs() < realmin, den.sign()*realmin, den)
    return xjprime / den_safe


# -----------------------------
# Faran FF (optional; gaussian default)
# -----------------------------
def faran_FF(a_um: float, freq_MHz: torch.Tensor, material: str = "glass", order: int = 25, normalize: bool = True):
    dev = freq_MHz.device; dt = freq_MHz.dtype
    c3, rho3 = 1540.0, 1.02
    ka = a_um * 2*math.pi * freq_MHz / c3
    if material.lower()=="glass":
        c1, rho1, nu = 5571.9, 2.38, 0.21
    elif material.lower()=="poly":
        c1, rho1, nu = 2000.0, 1.06, 0.35
    elif material.lower()=="fat":
        c1, rho1, nu = 1460.0, 0.94, 0.4993
    elif material.lower()=="tung":
        c1, rho1, nu = 5200.0, 19.3, 0.284
    elif material.lower()=="res":
        c1, rho1, nu = 1500.0, 0.90, 0.284
    elif material.lower()=="agar":
        c1, rho1, nu = 1500.0, 1.01, 0.4999
    else:
        c1, rho1, nu = 5571.9, 2.38, 0.21
    c2 = math.sqrt((0.5*c1*c1 - nu*c1*c1)/(1.0 - nu))
    x3 = ka; x1 = ka*(c3/c1); x2 = ka*(c3/c2)
    jx1 = sph_bessel_j(order, x1); jx2 = sph_bessel_j(order, x2); jx3 = sph_bessel_j(order, x3); yx3 = sph_bessel_y(order, x3)
    x1j1p = x_times_derivative_jn(jx1); x2j2p = x_times_derivative_jn(jx2); x3j3p = x_times_derivative_jn(jx3); x3y3p = x_times_derivative_jn(yx3)
    tiny = torch.tensor(1e-20, dtype=jx1.dtype, device=jx1.device)
    def sd(a, b): b2 = torch.where(b.abs()<tiny, torch.copysign(tiny, b), b); return a/b2
    delta_x3 = -sd(jx3, yx3); alpha_x3 = -sd(x3j3p, jx3); beta_x3 = -sd(x3y3p, yx3)
    r1 = x_over_j_times_jprime(jx1); r2 = x_over_j_times_jprime(jx2)
    alpha_x1 = -r1; alpha_x2 = -r2
    idx = torch.arange(0, order+1, device=jx1.device, dtype=jx1.dtype); n = idx.view(-1,1); i2 = n*(n+1.0)
    eps = torch.finfo(torch.float64).tiny
    d1 = torch.where((alpha_x1+1.0)==0, torch.copysign(torch.tensor(eps, dtype=alpha_x1.dtype, device=alpha_x1.device), alpha_x1), alpha_x1+1.0)
    S = n*(n+1.0); x2sq = x2*x2; b = 0.5*x2sq
    u = (alpha_x2 - 1.0 - b)/S
    one_plus_u = torch.where((1.0+u)==0, torch.copysign(torch.tensor(eps, dtype=u.dtype, device=u.device), u), 1.0+u)
    t1 = 1.0 - 1.0/d1; t2 = 1.0/one_plus_u; Num = t1 - t2
    t3 = 2.0 + (S - b - 2.0)/d1; t4 = (alpha_x2 + 1.0)/one_plus_u; Denom = t3 - t4
    tan_zeta = (-(x2**2)/2.0) * sd(Num, Denom)
    phi = -(rho3/rho1) * tan_zeta
    eta = torch.atan( delta_x3 * (alpha_x3 + phi) / (phi + beta_x3) )
    im = torch.complex(torch.tensor(0.0, device=dev), torch.tensor(1.0, device=dev))
    signs = ((-1.0)**idx).to(dt).view(-1,1)
    term = (2*n+1.0) * torch.sin(eta) * torch.exp(-im*eta) * signs
    s = term.sum(dim=0)
    k = x3 / (a_um*1e6)
    ff_raw = (s.abs() / torch.where(k.abs()<tiny, torch.full_like(k, tiny), k))**2
    return ff_raw


# -----------------------------
# FF theory matrix builder
# -----------------------------
def build_FF_theory(f_MHz: torch.Tensor, scat_diams_um: torch.Tensor, c: float,
                    model: str = "gaussian", use_faran: bool = False, material="glass"):
    F = f_MHz.shape[0]; D = scat_diams_um.shape[0]
    dev, dt = f_MHz.device, f_MHz.dtype
    FF = torch.zeros(F, D, device=dev, dtype=dt)
    if use_faran or (model.lower() == "faran"):
        for d in range(D):
            FF[:, d] = faran_FF((scat_diams_um[d].item()/2.0), f_MHz, material) * (f_MHz**4)
    else:
        # Gaussian: f^4 * exp(-0.827*(ka)^2), ka = pi*f*d/c
        for d in range(D):
            ka = math.pi * f_MHz * scat_diams_um[d] / c
            FF[:, d] = (f_MHz**4) * torch.exp(-0.827 * ka**2)
    return FF


def fit_ESD_EAC_from_logs(Lratio_dB_FR,          # [F,R] = 10log10(Sp/Sp_ref)
                          const_f_dB_F,          # [F]  = 10log10(BSC_ref) + 10log10(att_ratio)
                          FF_theory_FD,          # [F,D] (linear)
                          scat_diams_um, c):
    """
    Returns ESD_est[R], EAC_est[R], min_MSE[R] using fast MSE in dB.
    """
    dev = Lratio_dB_FR.device; F, R = Lratio_dB_FR.shape
    D = FF_theory_FD.shape[1]

    # Lconst_dB = const_f + Lratio
    Lc = Lratio_dB_FR + const_f_dB_F.view(F,1)   # [F,R]


    # precompute FF in dB & sums
    GdB = 10.0*torch.log10(torch.clamp(FF_theory_FD, 1e-30))  # [F,D]
    Sx  = Lc.sum(dim=0)                 # [R]
    Sx2 = (Lc*Lc).sum(dim=0)            # [R]
    Gy  = GdB.sum(dim=0)                # [D]
    Gy2 = (GdB*GdB).sum(dim=0)          # [D]
    Cross = Lc.t() @ GdB                # [R,D]

    Ff = float(F)
    # MSE(R,D) = (Sx2 - Sx^2/F) + (Gy2 - Gy^2/F) - 2*(Cross - Sx*Gy/F)
    MSE = (Sx2 - (Sx*Sx)/Ff).unsqueeze(1) \
        + (Gy2 - (Gy*Gy)/Ff).unsqueeze(0) \
        - 2.0*(Cross - (Sx.unsqueeze(1)*Gy.unsqueeze(0))/Ff)      # [R,D]


    min_MSE, idx = torch.min(MSE, dim=1)    # [R], [R]
    ESD_est = scat_diams_um[idx]            # [R]

    # --- EAC (linear mean, then dB) ---
    mTOcm, sTOus = 100.0, 1e6
    C_const = (math.pi**4) / (36.0 * ( (c*mTOcm/sTOus)**4 ))
    umTOcm = 1e-4
    d_cm6 = (ESD_est*umTOcm)**6             # [R]
    FF_pick = FF_theory_FD.index_select(1, idx)  # [F,R]
    Lin_ratio = torch.pow(10.0, Lratio_dB_FR/10.0)                # [F,R]
    EAC_lin = ( (torch.pow(10.0, const_f_dB_F/10.0).view(F,1) * Lin_ratio)
               / ( torch.clamp(FF_pick * (C_const * d_cm6.view(1,-1)), 1e-30) ) ).mean(dim=0)
    EAC_est = 10.0*torch.log10(EAC_lin.clamp_min(1e-30))  # [R]
    return ESD_est, EAC_est, min_MSE


import torch

@torch.no_grad()
def lratio_db_batch_stream(samB, refB, nz, nx, wz, wx, z0, x0, NFFT, idx, *, eps=1e-30):
    """
    10*log10(P_sam/P_ref) with explicit axial FFT on the last dim.
    samB, refB: complex [H, W, B]
    Returns: float32 [B, m, n, F]
    """
    dev = samB.device
    H, W, B = samB.shape
    m = int(z0.numel())
    n = int(x0.numel())
    Fbins = int(idx.numel())

    win_ax = torch.ones(nz, device=dev, dtype=torch.float32).view(nz, 1, 1)
    out = torch.empty(B, m, n, Fbins, device=dev, dtype=torch.float32)

    for ii in range(m):
        rstart = int(z0[ii].item())
        rstop  = rstart + nz
        if rstart < 0 or rstop > H:
            raise ValueError(f"Axial slice [{rstart}:{rstop}] out of range for H={H}")

        # [nz, W, B] with axial window
        slab_s = (samB[rstart:rstop, :, :]).mul(win_ax)
        slab_r = (refB[rstart:rstop, :, :]).mul(win_ax)

        # Lateral tiling (unfold along W): -> [nz, n, nx, B]
        view_s = slab_s.unfold(dimension=1, size=nx, step=wx)
        view_r = slab_r.unfold(dimension=1, size=nx, step=wx)

        # Reorder to [B, n, nx, nz] so axial is the last dim
        Vsb = view_s.permute(3, 1, 2, 0).contiguous()
        Vrb = view_r.permute(3, 1, 2, 0).contiguous()

        # Axial FFT on last dim, then fftshift on that dim
        Xs = torch.fft.fft(Vsb, n=NFFT, dim=3)
        Xr = torch.fft.fft(Vrb, n=NFFT, dim=3)

        Xs = torch.fft.fftshift(Xs, dim=3)
        Xr = torch.fft.fftshift(Xr, dim=3)

        # Power and average over lateral traces (nx) -> [B, n, NFFT]
        Ps_full = (Xs.real**2 + Xs.imag**2).mean(dim=0)
        Pr_full = (Xr.real**2 + Xr.imag**2).mean(dim=0)

        # Select desired bins (fftshifted)
        Ps = Ps_full.index_select(2, idx)  # [B, n, F]
        Pr = Pr_full.index_select(2, idx)
        Ps = Ps .permute(1, 0, 2)
        Pr = Pr.permute(1, 0, 2)
        # 10*log10 ratio
        L = 10.0 * (torch.log10(Ps.clamp_min(eps)) - torch.log10(Pr.clamp_min(eps)))
        out[:, ii, :, :] = L.to(torch.float32)

        del slab_s, slab_r, view_s, view_r, Vsb, Vrb, Xs, Xr, Ps_full, Pr_full, Ps, Pr, L

    return out  # [B, m, n, F]


@dataclass
class BSCPlan:
    # geometry / tiling
    device: str
    dtype: torch.dtype
    H_eff: int
    W_eff: int
    dx: float
    dz: float
    r: int
    wx: int
    nx: int
    wz: int
    nz: int
    m: int
    n: int
    x0: torch.Tensor
    z0: torch.Tensor
    # FFT band
    fs: float
    NFFT: int
    idx: torch.Tensor        # [F]
    f_base_MHz: torch.Tensor # [F]
    # reference constants & FF
    const_f_dB_F: torch.Tensor  # [F]
    scat_diams_um: torch.Tensor # [D]
    FF_theory_FD: torch.Tensor  # [F,D] linear
    c: float
    # fold cache
    fold: nn.Fold
    den_cache: torch.Tensor     # [1,1,H_eff,W_eff]


def _calc_dx_dz(H: int, W: int, device, dtype):
    z_m = torch.linspace(1, H, H, device=device, dtype=dtype) * (115.7e-3/H)
    x_m = torch.linspace(1, W, W, device=device, dtype=dtype) * (150e-3/W)
    dx = float((x_m[-1]-x_m[0])/(W-1))
    dz = float((z_m[-1]-z_m[0])/(H-1))
    return dx, dz


@torch.no_grad()
def build_bsc_plan(
    params: dict,
    bsc_tab_np: np.ndarray,
    freq_band_idx: np.ndarray,
    *,
    H: int = 552,
    W: int = 192,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> BSCPlan:
    dev = torch.device(device)

    # --- geometry / tiling (same rules as in esd_eac_batch) ---
    dx, dz = _calc_dx_dz(H, W, dev, dtype)
    wl_m = params.get("wl_m", 0.4e-3)
    blocksize_wl = params.get("blocksize_wl", 25)
    overlap_pc = params.get("overlap_pc", 0.875)
    r = round(1.0/(1.0 - overlap_pc))

    wx = round((blocksize_wl*wl_m)/(dx*r)); nx = r*wx
    nx = min(nx, W); wx = max(1, nx//r)

    ncol = math.floor((W-(r-1)*wx)/wx)
    W_eff = wx*(ncol+r-1)
    xi, xf = 0, W_eff-1
    x0 = torch.arange(xi, xf+1-nx+1, wx, device=dev)

    wz = math.floor(nx*dx/(dz*r)); nz = r*wz
    nrow = math.floor((H-(r-1)*wz)/wz)
    H_eff = wz*(nrow+r-1)
    zi, zf = 0, H_eff-1
    z0 = torch.arange(zi, zf+1-nz+1, wz, device=dev)

    n = int(x0.numel()); m = int(z0.numel())

    # --- FFT band (exact same selection logic) ---
    fs = params.get("fs", 5e6)
    NFFT = params.get("NFFT", 565)
    band = torch.linspace(-0.5, 0.5, NFFT, device=dev, dtype=dtype) * fs

    fL = params.get("freq_L_MHz", 3.003 - 3.6) * 1e6
    tol = 1e-2
    start_cands = torch.nonzero((band - fL) < tol, as_tuple=False).flatten()
    if start_cands.numel() == 0:
        start = int((band - fL).abs().argmin().item()) - 1
    else:
        start = int(start_cands[-1].item()) - 1
    start = max(0, start)

    low = int(params.get("freq_band_low_idx", 0))
    high = int(params.get("freq_band_high_idx", 115))
    Fref = high - low
    start = min(start, NFFT - Fref)
    idx = torch.arange(start, start + Fref, device=dev)

    f_base_MHz = (band.index_select(0, idx) * 1e-6)  # [F]

    # --- Reference BSC & const_f_dB ---
    BSC_ref = torch.from_numpy(bsc_tab_np[:,1].astype(np.float32)).to(dev)
    freq_ref = torch.from_numpy(bsc_tab_np[:,0].astype(np.float32)).to(dev)
    f1 = freq_ref[freq_band_idx]                 # as given
    BSC_ref_use = BSC_ref[freq_band_idx]

    # interpolate if needed to match f_base_MHz
    if f_base_MHz.numel() != f1.numel():
        BSC_ref_use = torch.from_numpy(
            np.interp(f_base_MHz.detach().cpu().numpy(),
                      f1.detach().cpu().numpy(),
                      BSC_ref_use.detach().cpu().numpy()).astype(np.float32)
        ).to(dev)
        f1 = f_base_MHz

    att_ratio = max(params.get("att_ref_dB",1.0) / max(params.get("att_sam_dB",1.0), 1e-12), 1e-12)
    const_f_dB_F = 10.0 * torch.log10(torch.clamp(BSC_ref_use, 1e-30))
    const_f_dB_F = const_f_dB_F + (10.0 * math.log10(att_ratio))  # [F]

    # --- FF theory (linear) ---
    scat_diams_um = torch.from_numpy(params.get("scat_diams_um")).to(dev).to(torch.float32)
    FF_theory_FD = build_FF_theory(
        f1, scat_diams_um, params.get("c",1540.0),
        model=params.get("form_factor","gaussian"),
        use_faran=params.get("use_faran", False),
        material=params.get("material", "glass")
    ).to(dtype)  # [F,D]

    # --- fold & denominator cache on device ---
    fold = nn.Fold(output_size=(H_eff, W_eff), kernel_size=(nz, nx), stride=(wz, wx)).to(dev)
    ones = torch.ones(1, nz * nx, m * n, device=dev, dtype=dtype)
    den_cache = fold(ones)  # [1,1,H_eff,W_eff]

    return BSCPlan(
        device=device, dtype=dtype, H_eff=H_eff, W_eff=W_eff,
        dx=dx, dz=dz, r=r, wx=wx, nx=nx, wz=wz, nz=nz, m=m, n=n,
        x0=x0, z0=z0,
        fs=fs, NFFT=NFFT, idx=idx, f_base_MHz=f_base_MHz,
        const_f_dB_F=const_f_dB_F,
        scat_diams_um=scat_diams_um, FF_theory_FD=FF_theory_FD,
        c=params.get("c",1540.0),
        fold=fold, den_cache=den_cache
    )


@torch.no_grad()
def lratio_db_batch_stream_vec(samB, refB, plan: BSCPlan, *, eps=1e-30):
    """
    Vectorized replacement for lratio_db_batch_stream.
    samB, refB: complex [H,W,B] (already on plan.device)
    Returns: float32 [B, m, n, F]
    """
    dev = samB.device
    H, W, B = samB.shape
    # crop to effective sizes that tile perfectly
    samB = samB[:plan.H_eff, :plan.W_eff, :]
    refB = refB[:plan.H_eff, :plan.W_eff, :]

    # Gather ALL axial tiles at once → [m, nz, W_eff, B]
    tiles_s = []
    tiles_r = []
    for i in range(plan.m):
        rstart = int(plan.z0[i].item())
        tiles_s.append(samB[rstart:rstart+plan.nz, :, :])  # [nz, W, B]
        tiles_r.append(refB[rstart:rstart+plan.nz, :, :])
    T_s = torch.stack(tiles_s, dim=0)  # [m, nz, W, B]
    T_r = torch.stack(tiles_r, dim=0)  # [m, nz, W, B]

    # Lateral sliding window via as_strided to avoid copies:
    # Turn each [m,nz,W,B] into [m,nz,n,nx,B]
    s0, s1, s2, s3 = T_s.stride()
    size = (plan.m, plan.nz, plan.n, plan.nx, B)
    stride = (s0, s1, s1*plan.wx, s1, s3)
    V_s = torch.as_strided(T_s, size=size, stride=stride)
    V_r = torch.as_strided(T_r, size=size, stride=stride)
    # Now permute to batch axial rows together: [B, m, n, nx, nz] → [B*m*n*nx, nz]
    V_s_b = V_s.permute(4,0,2,3,1).contiguous().view(B*plan.m*plan.n*plan.nx, plan.nz)
    V_r_b = V_r.permute(4,0,2,3,1).contiguous().view(B*plan.m*plan.n*plan.nx, plan.nz)

    # Axial FFT (one big batch), shift, power
    Xs = torch.fft.fft(V_s_b, n=plan.NFFT, dim=1)
    Xr = torch.fft.fft(V_r_b, n=plan.NFFT, dim=1)
    Xs = torch.fft.fftshift(Xs, dim=1)
    Xr = torch.fft.fftshift(Xr, dim=1)

    Ps = (Xs.real**2 + Xs.imag**2).view(B, plan.m, plan.n, plan.nx, plan.NFFT)
    Pr = (Xr.real**2 + Xr.imag**2).view(B, plan.m, plan.n, plan.nx, plan.NFFT)

    # Mean over nx then select desired bins
    Ps_mean = Ps.mean(dim=3).index_select(3, plan.idx)  # [B,m,n,F]
    Pr_mean = Pr.mean(dim=3).index_select(3, plan.idx)  # [B,m,n,F]

    L = 10.0 * (torch.log10(Ps_mean.clamp_min(eps)) - torch.log10(Pr_mean.clamp_min(eps)))
    return L.to(torch.float32)  # [B,m,n,F]


@torch.no_grad()
def esd_eac_batch_with_plan(IQ_obj_B: np.ndarray, IQ_ref_B: np.ndarray, plan: BSCPlan):
    """
    Same outputs as esd_eac_batch but reuses precomputed plan and vectorized FFT path.
    IQ_*_B: numpy complex64 [552,192,B]
    Returns: numpy float32 (esd_maps [B,552,192], eac_maps [B,552,192])
    """
    dev = torch.device(plan.device)
    # move data → device
    samB = torch.from_numpy(IQ_obj_B.astype(np.complex64)).to(dev)
    refB = torch.from_numpy(IQ_ref_B.astype(np.complex64)).to(dev)
    H0, W0, B = samB.shape

    # compute log-ratio [B,m,n,F]
    Lratio = try_lratio_with_backoff(samB, refB, plan)  # [B,m,n,F]
    Fbins = Lratio.shape[-1]
    R = B * plan.m * plan.n
    Lratio_FR = Lratio.permute(3,0,1,2).reshape(Fbins, R)  # [F,R]

    # Fit ESD/EAC using prebuilt const & FF_theory
    ESD_vec, EAC_vec, _ = fit_ESD_EAC_from_logs(
        Lratio_FR, plan.const_f_dB_F, plan.FF_theory_FD,
        plan.scat_diams_um, plan.c
    )  # [R]

    # reshape to [B,m,n]
    ESD = ESD_vec.view(B, plan.n, plan.m).transpose(1, 2).contiguous()
    EAC = EAC_vec.view(B, plan.n, plan.m).transpose(1, 2).contiguous()

    # ROI mask per slice (use plan sizes)
    IQ_abs = torch.abs(samB)  # [H,W,B]
    IQ_ds = F.interpolate(IQ_abs.permute(2,0,1).unsqueeze(1),
                          size=(plan.m, plan.n),
                          mode="bicubic", align_corners=False,
                          antialias=True).squeeze(1)                 # [B,m,n]
    thr = (0.02*IQ_ds.amax(dim=(1,2), keepdim=True))
    mask = (IQ_ds >= thr).to(ESD.dtype)

    ESD = (ESD * mask).contiguous()
    EAC = (EAC * mask).contiguous()

    # assemble to H_eff×W_eff via cached fold
    L = plan.m * plan.n
    patch_esd = ESD.view(B, 1, L).repeat(1, plan.nz*plan.nx, 1)
    patch_eac = EAC.view(B, 1, L).repeat(1, plan.nz*plan.nx, 1)

    num_esd = plan.fold(patch_esd)  # [B,1,H_eff,W_eff]
    num_eac = plan.fold(patch_eac)

    out_esd_eff = (num_esd / plan.den_cache.clamp_min(1e-12)).squeeze(1)  # [B,H_eff,W_eff]
    out_eac_eff = (num_eac / plan.den_cache.clamp_min(1e-12)).squeeze(1)

    # pad into full 552×192
    esd_maps = torch.zeros(B, 552, 192, device=dev, dtype=torch.float32)
    eac_maps = torch.zeros(B, 552, 192, device=dev, dtype=torch.float32)
    esd_maps[:, :plan.H_eff, :plan.W_eff] = torch.nan_to_num(out_esd_eff, nan=0.0, posinf=0.0, neginf=0.0)
    eac_maps[:, :plan.H_eff, :plan.W_eff] = torch.nan_to_num(out_eac_eff, nan=0.0, posinf=0.0, neginf=0.0)

    return esd_maps.detach().cpu().numpy(), eac_maps.detach().cpu().numpy()


def esd_eac_batch(IQ_obj_B: np.ndarray, IQ_ref_B: np.ndarray,
                  BSC_ref_tab: np.ndarray, freq_band_idx: np.ndarray,
                  params: dict, device="cuda", dtype=torch.float32):
    """
    Batched ESD/EAC for a stack of slices.
    IQ_obj_B, IQ_ref_B: numpy complex64 arrays [552,192,B]
    returns: (esd_maps [B,552,192], eac_maps [B,552,192])
    """
    dev = torch.device(device)
    samB = torch.from_numpy(IQ_obj_B.astype(np.complex64)).to(dev)
    refB = torch.from_numpy(IQ_ref_B.astype(np.complex64)).to(dev)
    H, W, B = samB.shape

    # geometry
    z_m = torch.linspace(1, 552, 552, device=dev) * (115.7e-3/552)
    x_m = torch.linspace(1, 192, 192, device=dev) * (150e-3/192)
    dx = float((x_m[-1]-x_m[0])/(W-1))
    dz = float((z_m[-1]-z_m[0])/(H-1))

    # sliding window
    wl_m = params.get("wl_m", 0.4e-3)
    blocksize_wl = params.get("blocksize_wl", 25)
    overlap_pc = params.get("overlap_pc", 0.875)
    r = round(1.0/(1.0 - overlap_pc))
    wx = round((blocksize_wl*wl_m)/(dx*r)); nx = r*wx
    nx = min(nx, W); wx = max(1, nx//r)

    # lateral tiling
    ncol = math.floor((W-(r-1)*wx)/wx)
    W_use = wx*(ncol+r-1)
    samB = samB[:, :W_use, :]
    refB = refB[:, :W_use, :]
    W = W_use
    xi, xf = 0, W-1
    x0 = torch.arange(xi, xf+1-nx+1, wx, device=dev)
    n = x0.numel()

    # axial tiling
    wz = math.floor(nx*dx/(dz*r)); nz = r*wz
    winsize = 0.5
    nw = int(max(3, 2*math.floor(winsize*nx*dx/(2*dz)) - 1))
    nrow = math.floor((H-(r-1)*wz)/wz)
    H_use = wz*(nrow+r-1)
    samB = samB[:H_use, :, :]
    refB = refB[:H_use, :, :]
    H = H_use
    zi, zf = 0, H-1
    z0 = torch.arange(zi, zf+1-nz+1, wz, device=dev)
    m = z0.numel()

    # FFT band
    fs = params.get("fs", 5e6)
    NFFT = params.get("NFFT", 565)
    band = torch.linspace(-0.5, 0.5, NFFT, device=dev) * fs  # Hz

    fL = params.get("freq_L_MHz", 3.003 - 3.6) * 1e6  # baseband Hz
    fH = params.get("freq_H_MHz", 6.0030 - 3.6) * 1e6  # baseband Hz

    low = int(params.get("freq_band_low_idx", 0))
    high = int(params.get("freq_band_high_idx", 115))
    Fref = high - low
    assert Fref > 0

    tol = 1e-2  # Hz
    start_cands = torch.nonzero((band - fL) < tol, as_tuple=False).flatten()
    if start_cands.numel() == 0:
        start = int((band - fL).abs().argmin().item()) - 1  # closest then minus one
    else:
        start = int(start_cands[-1].item()) - 1
    start = max(0, start)

    # take EXACTLY Fref bins; clamp if near the end
    start = min(start, NFFT - Fref)
    idx = torch.arange(start, start + Fref, device=dev)  # length == Fref
    f_base_MHz = (band[idx] * 1e-6)  # [Fref]
    # ---- log-ratio (dB) in ONE streamed pass ----
    Lratio = lratio_db_batch_stream(samB, refB, nz, nx, wz, wx, z0, x0, NFFT, idx)  # [B,m,n,F]
    Fbins = Lratio.shape[-1]
    R = B*m*n
    Lratio_FR = Lratio.permute(3,0,1,2).reshape(Fbins, R)                           # [F,R]

    # ---- reference BSC (interpolate if needed) ----
    BSC_ref = torch.from_numpy(BSC_ref_tab[:,1].astype(np.float32)).to(dev)
    freq_ref = torch.from_numpy(BSC_ref_tab[:,0].astype(np.float32)).to(dev)
    f1 = freq_ref[freq_band_idx]                     # [Fuse]
    BSC_ref_use = BSC_ref[freq_band_idx]            # [Fuse]
    if f_base_MHz.numel() != f1.numel():
        BSC_ref_use = torch.from_numpy(
            np.interp(f_base_MHz.cpu().numpy(), f1.cpu().numpy(), BSC_ref_use.cpu().numpy()
        ).astype(np.float32)).to(dev)
        f1 = f_base_MHz

    # const per frequency in dB: 10log10(BSC_ref) + 10log10(att_ref/att_sam)
    att_ratio = max(params.get("att_ref_dB",1.0) / max(params.get("att_sam_dB",1.0), 1e-12), 1e-12)
    const_f_dB = 10.0 * torch.log10(torch.clamp(BSC_ref_use, 1e-30))
    const_f_dB = const_f_dB + (10.0 * math.log10(att_ratio))

    # ---- FF theory (cache if you want) ----
    scat_diams_um = torch.from_numpy(params.get("scat_diams_um")).to(dev).to(torch.float32)
    FF_theory = build_FF_theory(f1, scat_diams_um, params.get("c",1540.0),
                                model=params.get("form_factor","gaussian"))  # [F,D]

    # ---- Fit (fast) ----
    ESD_vec, EAC_vec, minMSE = fit_ESD_EAC_from_logs(Lratio_FR, const_f_dB, FF_theory,
                                                     scat_diams_um, params.get("c",1540.0))  # [R]
    # reshape to [B,m,n]
    ESD = ESD_vec.view(B, n, m).transpose(1, 2).contiguous()  # -> [B, m, n]
    EAC = EAC_vec.view(B, n, m).transpose(1, 2).contiguous()  # -> [B, m, n]


    # ---- ROI mask per slice ----
    IQ_abs = torch.abs(samB)                                                   # [H,W,B]
    IQ_ds  = F.interpolate(IQ_abs.permute(2,0,1).unsqueeze(1),
                           size=(m,n), mode="bicubic", align_corners=False, antialias=True).squeeze(1)                 # [B,m,n]
    thr    = (0.02*IQ_ds.amax(dim=(1,2), keepdim=True))
    mask   = (IQ_ds >= thr).to(ESD.dtype)

    ESD = ESD * mask
    EAC = EAC * mask

    fold = nn.Fold(output_size=(H, W), kernel_size=(nz, nx), stride=(wz, wx)).to(dev)

    B, m, n = ESD.shape
    L = m * n

    # Denominator: pure overlap count (same as mask_for_overlap += 1 in MATLAB)
    ones = torch.ones(1, nz * nx, L, device=dev, dtype=torch.float32)
    den_cache = fold(ones)  # [1,1,H,W]

    # Numerators: masked tiles
    patch_esd = ESD.reshape(B, 1, L).repeat(1, nz * nx, 1)  # [B, nz*nx, L]
    num_esd = fold(patch_esd)  # [B,1,H,W]
    out_esd = (num_esd / den_cache).squeeze(1)  # [B,H,W]

    patch_eac = EAC.reshape(B, 1, L).repeat(1, nz * nx, 1)
    num_eac = fold(patch_eac)
    out_eac = (num_eac / den_cache).squeeze(1)

    # Final: pad and replace NaNs with 0 (exact MATLAB intent)
    esd_maps = torch.zeros(B, 552, 192, device=dev, dtype=torch.float32)
    eac_maps = torch.zeros(B, 552, 192, device=dev, dtype=torch.float32)
    esd_maps[:, :H, :W] = torch.nan_to_num(out_esd, nan=0.0, posinf=0.0, neginf=0.0)
    eac_maps[:, :H, :W] = torch.nan_to_num(out_eac, nan=0.0, posinf=0.0, neginf=0.0)

    return esd_maps.detach().cpu().numpy(), eac_maps.detach().cpu().numpy()



@torch.no_grad()
def lratio_db_batch_stream_vec_chunked(samB, refB, plan, nx_chunk=None, *, eps=1e-30):
    """
    VRAM-friendly log-ratio:
      - streams across the lateral nx dimension in chunks
      - accumulates mean over nx on the fly
    samB, refB: complex [H,W,B] on plan.device
    returns: float32 [B, m, n, F]
    """
    dev = samB.device
    H, W, B = samB.shape

    # Crop to perfectly tiling region
    samB = samB[:plan.H_eff, :plan.W_eff, :]
    refB = refB[:plan.H_eff, :plan.W_eff, :]

    # Gather axial tiles [m, nz, W, B] once
    tiles_s, tiles_r = [], []
    for i in range(plan.m):
        r0 = int(plan.z0[i].item())
        tiles_s.append(samB[r0:r0+plan.nz, :, :])  # [nz, W, B]
        tiles_r.append(refB[r0:r0+plan.nz, :, :])
    T_s = torch.stack(tiles_s, dim=0)  # [m, nz, W, B]
    T_r = torch.stack(tiles_r, dim=0)  # [m, nz, W, B]

    # Make a sliding view along W: [m, nz, n, nx, B] using as_strided (bounds-safe)
    s0, s1, s2, s3 = T_s.stride()      # strides for (m, nz, W, B)
    size   = (plan.m, plan.nz, plan.n, plan.nx, B)
    stride = (s0,     s1,      s2*plan.wx, s2,     s3)
    V_s = torch.as_strided(T_s, size=size, stride=stride)
    V_r = torch.as_strided(T_r, size=size, stride=stride)

    # Accumulators for mean over nx (we accumulate sums and divide by nx at end)
    Ps_acc = torch.zeros(B, plan.m, plan.n, plan.NFFT, device=dev, dtype=torch.float32)
    Pr_acc = torch.zeros_like(Ps_acc)

    # Choose chunk size
    if nx_chunk is None:
        nx_chunk = plan.nx  # try full; we’ll back off on OOM in the wrapper

    # Loop over nx in chunks
    for j0 in range(0, plan.nx, nx_chunk):
        j1 = min(j0 + nx_chunk, plan.nx)
        ch = j1 - j0

        # Slice the view to this chunk: [m, nz, n, ch, B] → batch axial rows
        Vs_c = V_s[:, :, :, j0:j1, :].permute(4, 0, 2, 3, 1) \
               .contiguous().view(B*plan.m*plan.n*ch, plan.nz)
        Vr_c = V_r[:, :, :, j0:j1, :].permute(4, 0, 2, 3, 1) \
               .contiguous().view(B*plan.m*plan.n*ch, plan.nz)

        # FFT along axial, shift, power
        Xs = torch.fft.fft(Vs_c, n=plan.NFFT, dim=1)
        Xr = torch.fft.fft(Vr_c, n=plan.NFFT, dim=1)
        Xs = torch.fft.fftshift(Xs, dim=1)
        Xr = torch.fft.fftshift(Xr, dim=1)

        # Reshape back and sum across this chunk of nx
        Psc = (Xs.real**2 + Xs.imag**2).view(B, plan.m, plan.n, ch, plan.NFFT).sum(dim=3)
        Prc = (Xr.real**2 + Xr.imag**2).view(B, plan.m, plan.n, ch, plan.NFFT).sum(dim=3)

        Ps_acc += Psc
        Pr_acc += Prc

        # free temps (helps GC under tight VRAM)
        del Vs_c, Vr_c, Xs, Xr, Psc, Prc

    # Mean over nx, then select passband, then dB ratio
    Ps_mean = (Ps_acc / float(plan.nx)).index_select(3, plan.idx)  # [B,m,n,F]
    Pr_mean = (Pr_acc / float(plan.nx)).index_select(3, plan.idx)
    L = 10.0 * (torch.log10(Ps_mean.clamp_min(eps)) - torch.log10(Pr_mean.clamp_min(eps)))
    return L.to(torch.float32)  # [B, m, n, F]


def try_lratio_with_backoff(samB, refB, plan, start_chunk=None):
    """
    Calls the chunked log-ratio and halves nx_chunk on CUDA OOM until it fits.
    Keeps B large and finds a VRAM-friendly nx_chunk automatically.
    """
    nx_chunk = start_chunk or plan.nx
    while True:
        try:
            return lratio_db_batch_stream_vec_chunked(samB, refB, plan, nx_chunk)
        except RuntimeError as e:
            msg = str(e)
            if ("CUDA out of memory" not in msg) or (nx_chunk <= 1):
                raise
            nx_chunk = max(1, nx_chunk // 2)
            torch.cuda.empty_cache()
            print(f"[VRAM] OOM -> reducing nx_chunk to {nx_chunk}")


def run_qt_bsc_single_slice(params, angle_lv: int, z_idx: int, *, write=False):
    dev = "cuda" if torch.cuda.is_available() and params.get("device","cuda")=="cuda" else "cpu"

    name = f"unit1_{angle_lv}"
    fref = os.path.join(params["ref_dir"], name)
    fobj = os.path.join(params["obj_dir"], name)
    with Tic(f"Angle {angle_lv} load (single slice)"):
        IQ_ref = read_iq_file_int32_interleaved(fref)   # [552,192,Nz]
        IQ_obj = read_iq_file_int32_interleaved(fobj)
    Nz = IQ_obj.shape[2]
    if not (0 <= z_idx < Nz):
        raise IndexError(f"z_idx {z_idx} out of range 0..{Nz-1}")

    # B=1 stacks
    refB = IQ_ref[:, :, z_idx:z_idx+1].astype(np.complex64, copy=False)
    objB = IQ_obj[:, :, z_idx:z_idx+1].astype(np.complex64, copy=False)

    # ref table & plan
    bsc_tab = np.loadtxt(params["BSC_ref_file"], delimiter=',').astype(np.float32)
    freq_band_idx = np.arange(params.get("freq_band_low_idx", 0),
                              params.get("freq_band_high_idx", 285), dtype=np.int64)

    plan = build_bsc_plan(params, bsc_tab, freq_band_idx, H=552, W=192, device=dev, dtype=torch.float32)

    # compute (compute-only timing)
    t0 = time.perf_counter()
    esdB, eacB = esd_eac_batch_with_plan(objB, refB, plan)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t_compute = time.perf_counter() - t0
    print(f"[TIME] Single-slice compute-only: {t_compute*1000:.1f} ms")

    esd_2d = esdB[0]; eac_2d = eacB[0]
    if write:
        out_esd = os.path.join(params["out_dir_esd"], f"{name}_z{z_idx:04d}")
        out_eac = os.path.join(params["out_dir_eac"], f"{name}_z{z_idx:04d}")
        write_int32_padded(out_esd, esd_2d[:, :, None])
        write_int32_padded(out_eac, eac_2d[:, :, None])
        print(f"[OK] wrote single-slice files:\n  {out_esd}\n  {out_eac}")
    return esd_2d, eac_2d, t_compute


def run_qt_bsc(params):
    dev = "cuda" if torch.cuda.is_available() and params.get("device","cuda")=="cuda" else "cpu"

    bsc_tab = np.loadtxt(params["BSC_ref_file"], delimiter=',').astype(np.float32)
    freq_band_idx = np.arange(params.get("freq_band_low_idx", 0),
                              params.get("freq_band_high_idx", 285), dtype=np.int64)

    os.makedirs(params["out_dir_esd"], exist_ok=True)
    os.makedirs(params["out_dir_eac"], exist_ok=True)

    for lv in range(1, 61):
        name = f"unit1_{lv}"
        fref = os.path.join(params["ref_dir"], name)
        fobj = os.path.join(params["obj_dir"], name)
        if not (os.path.exists(fref) and os.path.exists(fobj)):
            print(f"[WARN] missing angle {lv}, skipping"); continue

        with Tic(f"Angle {lv} load"):
            IQ_ref = read_iq_file_int32_interleaved(fref)   # [552,192,Nz]
            IQ_obj = read_iq_file_int32_interleaved(fobj)

        # Build plan once per angle (same geometry/params every batch)
        plan = build_bsc_plan(params, bsc_tab, freq_band_idx,
                              H=IQ_obj.shape[0], W=IQ_obj.shape[1],
                              device=dev, dtype=torch.float32)

        Nz = IQ_obj.shape[2]
        vol_esd = np.zeros((552, 192, Nz), dtype=np.float32)
        vol_eac = np.zeros((552, 192, Nz), dtype=np.float32)

        batch = int(params.get("batch_slices", 64))
        with Tic(f"Angle {lv} compute (batched)"):
            for z0 in range(0, Nz, batch):
                z1 = min(z0 + batch, Nz)
                objB = IQ_obj[:, :, z0:z1]
                refB = IQ_ref[:, :, z0:z1]
                esdB, eacB = esd_eac_batch_with_plan(objB, refB, plan)
                vol_esd[:, :, z0:z1] = np.transpose(esdB, (1, 2, 0))
                vol_eac[:, :, z0:z1] = np.transpose(eacB, (1, 2, 0))

        out_esd = os.path.join(params["out_dir_esd"], name)
        out_eac = os.path.join(params["out_dir_eac"], name)
        write_int32_padded(out_esd, vol_esd)
        write_int32_padded(out_eac, vol_eac)
        print(f"[OK] wrote {out_esd} and {out_eac}")


# -----------------------------
# Main (edit paths below)
# -----------------------------
if __name__ == "__main__":
    # ====== EDIT THESE PATHS ======
    params = dict(
        ref_dir = r"X:\ml132\Calculate_BSC_for_QT_big_phantom\BigPhantom\refl",
        obj_dir = r"Y:\Sunnybrook_patients_raw_data\QT002\QT002 RAW\da\refl",
        out_dir_esd = r"X:\ml132\GPU_QUS\BSCApp_for_paper\esd",
        out_dir_eac = r"X:\ml132\GPU_QUS\BSCApp_for_paper\eac",
        BSC_ref_file = r"X:\ml132\BSC_for_Sunnybrook\BigPhantom_BSC.txt",   # CSV with "MHz, value"
        # ====== Processing params ======
        wl_m = 0.4e-3,
        blocksize_wl = 25,
        overlap_pc = 0.875,
        att_ref_dB = 0.75,
        att_sam_dB = 0.75,
        fs = 5e6,
        freq_L_MHz = 3.5045-3.6,
        freq_H_MHz = 4.5074487-3.6,
        NFFT = 565,
        c = 1540.0,
        scat_diams_um = np.arange(5, 100+1, 2, dtype=np.float32),
        form_factor = "gaussian",
        use_faran=False,
        material="glass",
        device = "cuda",
        # reference BSC band indices
        freq_band_low_idx = 0,
        freq_band_high_idx = 115,
        batch_slices=8,
    )
    run_qt_bsc(params)

