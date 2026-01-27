# GPU_SLD.py
# ------------------------------------------------------------

# ------------------------------------------------------------
from __future__ import annotations
import os, math
from dataclasses import dataclass
from typing import Optional, Iterable, Tuple, Any
import time
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
import torch.nn as nn

from dataclasses import dataclass

@dataclass
class SLDPlan:
    device: str
    dtype: torch.dtype
    H: int
    W: int
    dx: float
    dz: float
    wx: int
    nx: int
    wz: int
    nz: int
    nw: int
    m: int
    n: int
    x0: torch.Tensor
    z0: torch.Tensor
    z0p: torch.Tensor
    z0d: torch.Tensor
    win: torch.Tensor
    win_half: int
    L_cm: float
    NFFT: int
    idx: torch.Tensor
    fMHz: torch.Tensor
    xvec: torch.Tensor
    fold: nn.Fold
    den_cache: torch.Tensor


def build_sld_plan(
    H: int, W: int, *,
    wl_m=0.4e-3, blocksize_wl=30, overlap_pc=0.875, fs=5e6,
    freq_L_MHz=-2.0, freq_H_MHz= 2.0, passband_shift_MHz=3.6,
    device="cuda", dtype=torch.float32,
    dx_m: Optional[float] = None,     # lateral spacing (m) per column
    dz_m: Optional[float] = None,     # axial spacing (m) per row
    x_extent_m: Optional[float] = None,  # total lateral span (m) across W-1 steps
    z_extent_m: Optional[float] = None,  # total axial   span (m) across H-1 steps
) -> SLDPlan:
    """
    If dx_m/dz_m are provided, they are used directly.
    Else if x_extent_m/z_extent_m are provided, spacings are derived as extent/(N-1).
    Else falls back to legacy QT spacings (~150mm lateral, ~115.7mm axial).
    """
    # --- resolve spacings (meters) ---
    if dx_m is not None and dz_m is not None:
        dx = float(dx_m)
        dz = float(dz_m)
    elif x_extent_m is not None and z_extent_m is not None:
        dx = float(x_extent_m) / max(1, (W - 1))
        dz = float(z_extent_m) / max(1, (H - 1))
    else:
        # Legacy QT defaults: 150mm lateral, 115.7mm axial across image
        dx = (150e-3) / max(1, (W - 1))
        dz = (115.7e-3) / max(1, (H - 1))

    # --- lateral tiling ---
    r  = round(1.0 / (1.0 - overlap_pc))
    wx = round((blocksize_wl * wl_m) / (dx * r))
    nx = max(1, r * wx); nx = min(nx, W)
    wx = max(1, nx // r)

    # make width tile perfectly
    L2_used = wx * (math.floor((W - (r - 1) * wx) / wx) + r - 1)
    W_eff = min(W, L2_used)
    xi, xf = 0, W_eff - 1
    x0 = torch.arange(xi, xf + 1 - nx + 1, wx, device=device)
    n  = x0.numel()

    # --- axial tiling ---
    wz = math.floor(nx * dx / (dz * r))
    nz = max(3, r * wz); nz = min(nz, H)
    wz = max(1, nz // r)

    # axial *window* (nw) within each nz-tall tile
    winsize = 0.5
    nw = 2 * math.floor(winsize * nx * dx / (2 * dz)) - 1
    nw = int(max(3, min(nw, nz - 2)))
    win = torch.ones(nw, device=device, dtype=dtype)
    win_half = (nw - 1) // 2

    # path length between proximal/distal windows [cm]
    L_cm = (nz - nw) * dz * 100.0

    # --- frequency grid & passband (fs must be the POST-decimation sampling rate) ---
    NFFT = 1 << (int(math.ceil(math.log2(max(8, nw)))) + 1)
    band = torch.linspace(-0.5, 0.5, NFFT, device=device, dtype=dtype) * fs
    fL = freq_L_MHz * 1e6
    fH = freq_H_MHz * 1e6
    start = torch.argmin((band - fL).abs()).item()
    end   = torch.argmin((band - fH).abs()).item()
    if start > end:
        start, end = end, start
    idx  = torch.arange(start, end, device=device)
    fMHz = (band.index_select(0, idx) * 1e-6) + passband_shift_MHz

    # --- axial positions (tile starts) ---
    L1_used = wz * (math.floor((H - (r - 1) * wz) / wz) + r - 1)
    H_eff = min(H, L1_used)
    zi, zf = 0, H_eff - 1
    z0  = torch.arange(zi, zf + 1 - nz + 1, wz, device=device)
    m   = z0.numel()
    z0p = z0 + win_half
    z0d = z0 + (nz - 1) - win_half

    # --- fold kernel & denominator cache ---
    fold = nn.Fold(output_size=(H_eff, W_eff), kernel_size=(nz, nx), stride=(wz, wx)).to(device)
    ones = torch.ones(1, nz * nx, m * n, device=device, dtype=dtype)
    den_cache = fold(ones)  # [1,1,H_eff,W_eff]

    # LS x-vector
    xvec = (4.0 * L_cm * fMHz).to(dtype=dtype)  # [L3]

    return SLDPlan(
        device, dtype, H_eff, W_eff, dx, dz, wx, nx, wz, nz, nw, m, n,
        x0, z0, z0p, z0d, win, win_half, L_cm, NFFT, idx, fMHz, xvec, fold, den_cache
    )


# --- batched kernel that reuses the plan (no geometry/FFT setup per batch) ---
def sld_angle_fast_batch_with_plan(IQ_obj_batch, IQ_ref_batch, plan: SLDPlan, *,
                                   att_ref_dB_per_cm_MHz=1.0, eps=1e-12):
    """
    IQ_*_batch: complex [H,W,B] on plan.device. H/W may be >= plan.H/W; we crop to plan.H/W.
    Returns: float32 [B, plan.H, plan.W] (caller pads to 552x192 if needed).
    """
    device, dtype = plan.device, plan.dtype
    H, W, B = IQ_obj_batch.shape

    # crop to effective sizes so windows tile perfectly
    Obj = IQ_obj_batch[:plan.H, :plan.W, :]
    Ref = IQ_ref_batch[:plan.H, :plan.W, :]

    # Gather blocks (uses *nw* rows per window), loop over m (small), vectorized over n,nx,B
    def gather_blocks_nw(vol, centers):  # vol: [H,W,B] complex
        H, W, B = vol.shape
        blocks = []
        r0 = (centers - plan.win_half).clamp(0, H - plan.nw)
        for i in range(plan.m):
            rstart = int(r0[i].item())
            slab = vol[rstart:rstart + plan.nw, :, :]          # [nw, W, B]
            # lateral strided view -> [nw, n, nx, B]
            s0, s1, s2 = slab.stride()
            view = torch.as_strided(
                slab,
                size=(plan.nw, plan.n, plan.nx, B),
                stride=(s0, s1 * plan.wx, s1, s2)
            )  # [nw, n, nx, B]
            blocks.append(view)
        T = torch.stack(blocks, dim=0)                          # [m, nw, n, nx, B]
        return T

    Sp_p = gather_blocks_nw(Obj, plan.z0p) * plan.win.view(1, plan.nw, 1, 1, 1)
    Sp_d = gather_blocks_nw(Obj, plan.z0d) * plan.win.view(1, plan.nw, 1, 1, 1)
    Sr_p = gather_blocks_nw(Ref, plan.z0p) * plan.win.view(1, plan.nw, 1, 1, 1)
    Sr_d = gather_blocks_nw(Ref, plan.z0d) * plan.win.view(1, plan.nw, 1, 1, 1)

    # FFT on axial dim (nw -> NFFT), batched over (B*m*n*nx)
    def power(blocks):  # [m, nw, n, nx, B] complex
        m, nw, n, nx, B = blocks.shape
        X = torch.fft.fft(blocks.permute(4,0,2,3,1).reshape(B*m*n*nx, nw),
                          n=plan.NFFT, dim=1)
        X = torch.fft.fftshift(X, dim=1)
        P = (X.abs()**2).view(B, m, n, nx, plan.NFFT)  # [B,m,n,nx,NFFT]
        Psel = P.index_select(4, plan.idx)             # [B,m,n,nx,L3]
        return Psel.mean(dim=3)                        # mean over nx -> [B,m,n,L3]

    Sp    = power(Sp_p)
    Sd    = power(Sp_d)
    Spref = power(Sr_p)
    Sdref = power(Sr_d)

    att_ref = (att_ref_dB_per_cm_MHz/8.686) * plan.fMHz             # [L3]
    diff_comp = (torch.log(Spref+eps) - torch.log(Sdref+eps)) - 4.0*plan.L_cm*att_ref
    y = (torch.log(Sp+eps) - torch.log(Sd+eps)) - diff_comp         # [B,m,n,L3]

    p    = y.shape[-1]
    Sx   = plan.xvec.sum()
    Sxx  = (plan.xvec*plan.xvec).sum()
    Sy   = y.sum(dim=-1)                                            # [B,m,n]
    Sxy  = (y * plan.xvec.view(1,1,1,-1)).sum(dim=-1)               # [B,m,n]
    denom = p*Sxx - Sx*Sx + 1e-12
    beta  = (p*Sxy - Sx*Sy)/denom                                   # [B,m,n]
    BSdB  = 8.686 * beta

    # ROI mask per slice (reuse plan sizes)
    IQ_abs = Obj.abs().permute(2,0,1).unsqueeze(1)                  # [B,1,H,W]
    IQ_ds  = F.interpolate(IQ_abs, size=(plan.m, plan.n),
                           mode="bicubic", align_corners=False, antialias=True).squeeze(1)
    thr    = (0.02*IQ_ds.amax(dim=(1,2), keepdim=True))
    mask   = (IQ_ds >= thr).to(BSdB.dtype)
    BSdB   = (BSdB * mask).contiguous()                             # [B,m,n]

    # assemble via prebuilt fold + den_cache (tile kernel uses nz*nx)
    patches = BSdB.reshape(B, 1, plan.m*plan.n).repeat(1, plan.nz*plan.nx, 1)  # [B, nz*nx, L]
    num = plan.fold(patches)                                        # [B,1,H_eff,W_eff]
    out = (num / plan.den_cache.clamp_min(1e-12)).squeeze(1)        # [B,H_eff,W_eff]
    return out


def sld_angle_fast_batch_with_plan_grid(IQ_obj_batch, IQ_ref_batch, plan: SLDPlan, *,
                                        att_ref_dB_per_cm_MHz=1.0, eps=1e-12,
                                        apply_mask=False, roi_percent=0.02, pool="max"):
    """
    Grid-only variant (NO fold/upsample). Returns [B, m, n] directly.
      - apply_mask=False (default) disables ROI masking entirely.
      - If apply_mask=True, downsample magnitude to (m,n) via adaptive pool
        (no F.interpolate), build a coarse mask, and multiply.

    IQ_*_batch: complex [H,W,B]
    Returns: float32 [B, plan.m, plan.n]
    """
    # --- crop to effective sizes so windows tile perfectly ---
    Obj = IQ_obj_batch[:plan.H, :plan.W, :]
    Ref = IQ_ref_batch[:plan.H, :plan.W, :]

    # --- gather blocks (same as your original) ---
    def gather_blocks_nw(vol, centers):  # vol: [H,W,B] complex
        H_, W_, B_ = vol.shape
        blocks = []
        r0 = (centers - plan.win_half).clamp(0, H_ - plan.nw)
        for i in range(plan.m):
            rstart = int(r0[i].item())
            slab = vol[rstart:rstart + plan.nw, :, :]          # [nw, W, B]
            s0, s1, s2 = slab.stride()
            view = torch.as_strided(
                slab,
                size=(plan.nw, plan.n, plan.nx, B_),
                stride=(s0, s1 * plan.wx, s1, s2)
            )  # [nw, n, nx, B]
            blocks.append(view)
        T = torch.stack(blocks, dim=0)                          # [m, nw, n, nx, B]
        return T

    Sp_p = gather_blocks_nw(Obj, plan.z0p) * plan.win.view(1, plan.nw, 1, 1, 1)
    Sp_d = gather_blocks_nw(Obj, plan.z0d) * plan.win.view(1, plan.nw, 1, 1, 1)
    Sr_p = gather_blocks_nw(Ref, plan.z0p) * plan.win.view(1, plan.nw, 1, 1, 1)
    Sr_d = gather_blocks_nw(Ref, plan.z0d) * plan.win.view(1, plan.nw, 1, 1, 1)

    # --- FFT & power ---
    def power(blocks):  # [m, nw, n, nx, B] complex
        m, nw, n, nx, B = blocks.shape
        X = torch.fft.fft(blocks.permute(4,0,2,3,1).reshape(B*m*n*nx, nw),
                          n=plan.NFFT, dim=1)
        X = torch.fft.fftshift(X, dim=1)
        P = (X.abs()**2).view(B, m, n, nx, plan.NFFT)  # [B,m,n,nx,NFFT]
        Psel = P.index_select(4, plan.idx)             # [B,m,n,nx,L3]
        return Psel.mean(dim=3)                        # -> [B,m,n,L3]

    Sp    = power(Sp_p)
    Sd    = power(Sp_d)
    Spref = power(Sr_p)
    Sdref = power(Sr_d)

    # --- LS fit (same as original) -> BSdB [B,m,n] ---
    att_ref = (att_ref_dB_per_cm_MHz/8.686) * plan.fMHz             # [L3]
    diff_comp = (torch.log(Spref+eps) - torch.log(Sdref+eps)) - 4.0*plan.L_cm*att_ref
    y = (torch.log(Sp+eps) - torch.log(Sd+eps)) - diff_comp         # [B,m,n,L3]

    p    = y.shape[-1]
    Sx   = plan.xvec.sum()
    Sxx  = (plan.xvec*plan.xvec).sum()
    Sy   = y.sum(dim=-1)                                            # [B,m,n]
    Sxy  = (y * plan.xvec.view(1,1,1,-1)).sum(dim=-1)               # [B,m,n]
    denom = p*Sxx - Sx*Sx + 1e-12
    beta  = (p*Sxy - Sx*Sy)/denom                                   # [B,m,n]
    BSdB  = (8.686 * beta).contiguous()                              # [B,m,n]

    # --- optional ROI mask on the grid (no interpolate; uses adaptive pool) ---
    if apply_mask and roi_percent and roi_percent > 0:
        IQ_abs = Obj.abs().permute(2,0,1).unsqueeze(1)  # [B,1,H,W]
        if pool == "max":
            IQ_ds = torch.nn.functional.adaptive_max_pool2d(IQ_abs, (plan.m, plan.n)).squeeze(1)
        else:
            IQ_ds = torch.nn.functional.adaptive_avg_pool2d(IQ_abs, (plan.m, plan.n)).squeeze(1)
        thr  = roi_percent * IQ_ds.amax(dim=(1,2), keepdim=True)
        mask = (IQ_ds >= thr).to(BSdB.dtype)
        BSdB = (BSdB * mask)

    return BSdB  # [B, m, n]

# ===================== Editable CONFIG =====================


# --- MAT mode config (per-angle RF 2D .mat slices + single reference.mat) ---
MAT_CONFIG = {
    "obj_dir": r"Z:\ml132\GPU_QUS\sim_atten_data",  # folder with mat (object)
    "ref_file": r"Z:\ml132\GPU_QUS\sim_atten_data\reference.mat",
    "out_dir": r"Z:\ml132\GPU_QUS\gpu_attenuation_from_mat",
    "var_obj": "rf",  # variable name inside .mat (object RF)
    "var_ref": "rf2",  # variable name inside .mat (reference RF)
    "pattern": "angle_{:03d}.mat",  # filename pattern
    "angles": list(range(1, 61)),  # which angles to process

    # ---- Pre-RF2IQ cropping (MATLAB-style inclusive ranges) ----
    # example: rows 1760:5279, cols 64:191
    # Set to None to disable cropping.
    "crop_rows_matlab": (1760, 5279),
    "crop_cols_matlab": (64, 191),

    # RF->IQ params
    "Fs": 5.114035054971530e7,  # sampling rate (Hz)
    "Fc": 3.6e6,  # center freq (Hz). Put a number (e.g., 5e6) if known; None => auto-estimate
    "Bpct": None,  # fractional bandwidth in % (None => default min(2*Fc, Fs/2))
    "decim": 8,  # integer decimation factor after LPF (>=1)
    "axial_dim": 0,  # fast-time axis in your RF arrays (if RF is [H,W,Na], axial is usually 0)
    "taps": 129,  # symmetric FIR taps (odd)

    # SLD/plan params
    "wl_m": 0.38e-3,
    "blocksize_wl": 40,
    "overlap_pc": 0.875,
    "fs_proc": 6.392543818714412e06,
    "freq_L_MHz": 2.5 - 3.6,
    "freq_H_MHz": 5.5 - 3.6,
    "passband_shift_MHz": 3.6,
    "att_ref": 0.2,  # dB/cm/MHz
    "batch_slices": 60,
    # MAT_CONFIG additions
    "resize_to_input": False,          # resize SLD map back to the RF→IQ input H×W (after decimation)
    "restore_pre_decim": False,       # if True, upsample again by 'decim' along the axial dim to pre-decim size
    "resize_to_iq": True,

    # Single frame mode
    "input_mode": "single",         # "stack" (default) or "single"
    "input_kind": "RF",            # "RF" (will RF->IQ) or "IQ" (already complex baseband)
    "single_obj_file": r"Z:\ml132\GPU_QUS\sim_atten_data\angle_001.mat",  # used only when input_mode="single"
    "single_ref_file": r"Z:\ml132\GPU_QUS\sim_atten_data\reference.mat",  # used only when input_mode="single"
    "single_var_obj": "rf",        # variable name/path for object in the single .mat
    "single_var_ref": "rf2",       # variable name/path for reference in the single .mat
    "single_out_name": "atten_single",
    "single_write_2d": True,       # True => save [H,W] (squeezed). False => save [H,W,1]

    # Output layout
    "output_layout": "native",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": "float32",             # "float32", "float16", "bfloat16"
}

# ===================== Types & small utilities =====================

_DTYPE_MAP = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

@dataclass
class RF2IQParams:
    Fs: float
    Fc: Optional[float]
    Bpct: Optional[float]
    decim: int
    axial_dim: int
    taps: int

@dataclass
class RunParams:
    output_layout: str
    wl_m: float
    blocksize_wl: int
    overlap_pc: float
    fs_proc: float
    freq_L_MHz: float
    freq_H_MHz: float
    passband_shift_MHz: float
    att_ref: float
    batch_slices: int
    device: str
    dtype: torch.dtype

def _rp_from_dict(d: dict) -> RunParams:
    return RunParams(
        output_layout=d.get("output_layout", "qt1104x192"),
        wl_m=d["wl_m"],
        blocksize_wl=d["blocksize_wl"],
        overlap_pc=d["overlap_pc"],
        fs_proc=d["fs_proc"],
        freq_L_MHz=d["freq_L_MHz"],
        freq_H_MHz=d["freq_H_MHz"],
        passband_shift_MHz=d["passband_shift_MHz"],
        att_ref=d["att_ref"],
        batch_slices=d["batch_slices"],
        device=d.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        dtype=_DTYPE_MAP[d.get("dtype", "float32")],
    )

def _rf2iqp_from_dict(d: dict) -> RF2IQParams:
    return RF2IQParams(
        Fs=d["Fs"], Fc=d.get("Fc"), Bpct=d.get("Bpct"),
        decim=int(d.get("decim", 1)), axial_dim=int(d.get("axial_dim", 0)),
        taps=int(d.get("taps", 129)),
    )

# ----- MATLAB-style range (inclusive) -> Python slice (0-based, end-exclusive)
def _matlab_range_to_slice(rr: Tuple[int,int]) -> slice:
    start_m, end_m = rr
    if not (isinstance(start_m, int) and isinstance(end_m, int) and start_m >= 1 and end_m >= start_m):
        raise ValueError(f"Invalid MATLAB range: {rr}")
    # MATLAB 1-based inclusive -> Python 0-based slice
    return slice(start_m - 1, end_m)


def _apply_optional_crop_2d(a: np.ndarray, rows_matlab: Optional[Tuple[int,int]], cols_matlab: Optional[Tuple[int,int]]) -> np.ndarray:
    if rows_matlab is not None:
        rs = _matlab_range_to_slice(rows_matlab)
    else:
        rs = slice(None)
    if cols_matlab is not None:
        cs = _matlab_range_to_slice(cols_matlab)
    else:
        cs = slice(None)
    return a[rs, cs]


# ----- Resolve variable from .mat dict, supporting dotted path for structs
def _resolve_mat_var(mat: dict, var: str) -> np.ndarray:
    """
    Supports:
      - plain array: var="RF_obj"
      - struct path: var="ref1.rf2" (will traverse fields)
    Uses squeeze to simplify dims.
    """
    # load with squeeze so structs are matlab objects
    # (callers in this file pass data already loaded with squeeze)
    parts = var.split(".")
    cur: Any = mat
    for i, p in enumerate(parts):
        # dict key
        if isinstance(cur, dict):
            cur = cur[p]
        else:
            # MATLAB struct via scipy -> possibly mat_struct; try attribute access
            if hasattr(cur, p):
                cur = getattr(cur, p)
            elif isinstance(cur, np.ndarray) and cur.dtype.names and p in cur.dtype.names:
                cur = cur[p]
            else:
                raise KeyError(f"Cannot resolve field '{p}' in path '{var}'")
        # squeeze one level if it's singleton arrays
        if isinstance(cur, np.ndarray) and cur.size == 1 and cur.shape != ():
            cur = np.squeeze(cur)
    # finally to numpy array
    return np.array(cur)


def decimate_axial_every_M(x: torch.Tensor, M: int, axial_dim: int = 0) -> torch.Tensor:
    """
    Keep every M-th sample along axial_dim. No filtering (mirrors MATLAB downsample()).
    """
    if M <= 1:
        return x
    idx = torch.arange(0, x.shape[axial_dim], M, device=x.device)
    return x.index_select(axial_dim, idx)

import torch, torch.nn.functional as F, math
import numpy as np
from scipy.signal import butter, filtfilt


def rf2iq_downmix_lpf_butter_cpu(rf_np: np.ndarray, *, Fs: float, Fc: float,
                                 Bpct: float | None = None, axial_axis: int = 0) -> np.ndarray:
    """
    CPU version using scipy.signal.butter + filtfilt for MATLAB parity.
    rf_np: real numpy array
    returns complex64 numpy array, same shape
    """
    rf = np.moveaxis(rf_np, axial_axis, -1)
    L  = rf.shape[-1]
    t  = np.arange(L, dtype=np.float64) / float(Fs)
    z  = rf.astype(np.complex64) * np.exp(-1j*2*np.pi*Fc*t).astype(np.complex64)

    # Normalized cutoff Wn
    if Bpct is None:
        Wn = min(2.0*Fc/Fs, 0.5)
    else:
        Wn = (Bpct*Fc/100.0) / Fs
    b, a = butter(5, Wn, btype='low', analog=False)

    # filtfilt real & imag separately
    zr = filtfilt(b, a, z.real, axis=-1) * 2.0
    zi = filtfilt(b, a, z.imag, axis=-1) * 2.0
    zf = (zr + 1j*zi).astype(np.complex64)

    return np.moveaxis(zf, -1, axial_axis)


def rf2iq_downmix_lpf(rf: torch.Tensor, *, Fs: float, Fc: float,
                      Bpct: float | None = None, taps: int = 129,
                      axial_dim: int = 0) -> torch.Tensor:
    """
    RF->IQ (downmix + LPF) like MATLAB rf2iq, but NO decimation.
    Uses symmetric windowed-sinc FIR with reflect padding (zero-phase effect).
    rf: real tensor [...], axial axis = axial_dim
    returns complex tensor same shape as rf
    """
    dev, dt = rf.device, rf.dtype
    # Cutoff: default Wn = min(2*Fc/Fs, 0.5) -> cutoff_Hz = min(2*Fc, Fs/2)
    cutoff_Hz = min(2.0*Fc, 0.5*Fs) if Bpct is None else (Bpct*Fc)/100.0

    # Move axial to last
    rf_m = rf.moveaxis(axial_dim, -1)
    L = rf_m.shape[-1]
    t = torch.arange(L, device=dev, dtype=dt) / float(Fs)
    ph = torch.exp(-1j * 2.0 * math.pi * float(Fc) * t)
    z  = rf_m.to(torch.complex64) * ph  # downmix

    # FIR low-pass
    n = torch.arange(taps, device=dev, dtype=rf.real.dtype) - (taps-1)/2
    wc = 2.0*math.pi*(cutoff_Hz/Fs)
    h  = (wc/math.pi) * torch.sinc((wc/math.pi)*n)
    w  = 0.5*(1.0 - torch.cos(2.0*math.pi*(torch.arange(taps, device=dev, dtype=rf.real.dtype)/(taps-1))))
    h  = (h*w); h = h / h.sum().clamp_min(1e-30)
    pad = (taps-1)//2
    k = h.flip(0).view(1,1,-1)

    zr = z.real.reshape(-1,1,L)
    zi = z.imag.reshape(-1,1,L)
    zr = F.conv1d(F.pad(zr, (pad,pad), mode='reflect'), k)
    zi = F.conv1d(F.pad(zi, (pad,pad), mode='reflect'), k)
    zf = torch.complex(zr.squeeze(1), zi.squeeze(1)).reshape(rf_m.shape) * 2.0

    return zf.moveaxis(-1, axial_dim).contiguous()  # complex


# ===================== MAT mode helpers =====================
def load_single_mat_with_ref(
    obj_file: str,
    ref_file: str,
    *,
    var_obj: str,
    var_ref: str,
    crop_rows_matlab: Optional[Tuple[int,int]],
    crop_cols_matlab: Optional[Tuple[int,int]],
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load a SINGLE 2D frame for object and reference, apply same crop, and
    return tensors shaped [H,W,1] so the downstream pipeline stays unchanged.
    """
    mobj = scipy.io.loadmat(obj_file, struct_as_record=False, squeeze_me=True)
    mref = scipy.io.loadmat(ref_file, struct_as_record=False, squeeze_me=True)

    Obj2d = _resolve_mat_var(mobj, var_obj)
    Ref2d = _resolve_mat_var(mref, var_ref)

    if Obj2d.ndim != 2:
        raise ValueError(f"single object must be 2D, got {Obj2d.shape}")
    if Ref2d.ndim != 2:
        raise ValueError(f"single reference must be 2D, got {Ref2d.shape}")

    Obj2d = _apply_optional_crop_2d(Obj2d, crop_rows_matlab, crop_cols_matlab)
    Ref2d = _apply_optional_crop_2d(Ref2d, crop_rows_matlab, crop_cols_matlab)

    if Obj2d.shape != Ref2d.shape:
        raise ValueError(f"object {Obj2d.shape} and reference {Ref2d.shape} must match")

    RFo = torch.from_numpy(Obj2d[:, :, None]).to(device)  # [H,W,1]
    RFr = torch.from_numpy(Ref2d[:, :, None]).to(device)
    return RFo, RFr


def load_mat_angles_stack_with_ref(
    obj_dir: str,
    ref_file: str,
    *,
    var_obj: str = "RF_obj",
    var_ref: str = "RF_ref",
    pattern: str = "angle_{:03d}.mat",
    angles: Iterable[int] = range(1,61),
    crop_rows_matlab: Optional[Tuple[int,int]] = None,
    crop_cols_matlab: Optional[Tuple[int,int]] = None,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load per-angle OBJECT RF 2D slices -> stack to [H,W,Na].
    Load single REFERENCE RF 2D -> crop same way -> tile to [H,W,Na].
    Supports dotted var names (structs).
    """
    RFo_list = []
    H = W = None

    # Load angles
    for lv in angles:
        mobj = scipy.io.loadmat(os.path.join(obj_dir, pattern.format(lv)), struct_as_record=False, squeeze_me=True)
        RFo2d = _resolve_mat_var(mobj, var_obj)  # expect 2D
        if RFo2d.ndim != 2:
            raise ValueError(f"[angle {lv}] need 2D OBJECT array; got {RFo2d.shape}")

        # crop before stacking
        RFo2d = _apply_optional_crop_2d(RFo2d, crop_rows_matlab, crop_cols_matlab)

        if H is None:
            H, W = RFo2d.shape
        else:
            if RFo2d.shape != (H,W):
                raise ValueError(f"Inconsistent OBJECT shape at angle {lv}: {RFo2d.shape} vs {(H,W)}")

        RFo_list.append(RFo2d)

    RFo = torch.from_numpy(np.stack(RFo_list, axis=2)).to(device)  # [H,W,Na]

    # Load reference once and crop the same way
    mref = scipy.io.loadmat(ref_file, struct_as_record=False, squeeze_me=True)
    RFr2d = _resolve_mat_var(mref, var_ref)  # expect 2D
    if RFr2d.ndim != 2:
        raise ValueError(f"reference.mat var '{var_ref}' must be 2D; got {RFr2d.shape}")
    RFr2d = _apply_optional_crop_2d(RFr2d, crop_rows_matlab, crop_cols_matlab)

    if RFr2d.shape != (H,W):
        raise ValueError(f"reference.mat cropped shape {RFr2d.shape} != expected {(H,W)}")

    # tile along angles
    RFr = np.repeat(RFr2d[:, :, None], RFo.shape[2], axis=2)
    RFr = torch.from_numpy(RFr).to(device)  # [H,W,Na]

    return RFo, RFr

def write_native_mat(outB: torch.Tensor, fname: str, var: str = "atten"):
    B,H,W = outB.shape
    arr = outB.detach().cpu().numpy().transpose(1,2,0)  # [H,W,B]
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    scipy.io.savemat(fname + ".mat", {var: arr})

# ===================== Core: run SLD on a stack =====================

def run_sld_on_stack_nopad(IQo: torch.Tensor, IQr: torch.Tensor, rp: RunParams) -> torch.Tensor:
    """
    IQo/IQr: complex [H, W, B]
    Returns: float32 [B, plan.H, plan.W]  (NO padding; just the effective region)
    """
    H, W, B = IQo.shape
    kgrid_dt = 1.9554031e-08
    c0 = 1540.0
    dz_m = c0 * (kgrid_dt * 8) / 2.0  # axial spacing (meters)
    dx_m = (1060 * 1.0038e-04) / (256 - 1)  # lateral spacing (meters)

    plan = build_sld_plan(
        H=IQo.shape[0], W=IQo.shape[1],
        wl_m=rp.wl_m, blocksize_wl=rp.blocksize_wl, overlap_pc=rp.overlap_pc,
        fs=rp.fs_proc,
        freq_L_MHz=rp.freq_L_MHz, freq_H_MHz=rp.freq_H_MHz, passband_shift_MHz=rp.passband_shift_MHz,
        device=rp.device, dtype=rp.dtype,
        dx_m=dx_m, dz_m=dz_m,
    )

    IQo = IQo.to(rp.device, non_blocking=True)
    IQr = IQr.to(rp.device, non_blocking=True)
    out_eff = sld_angle_fast_batch_with_plan_grid(
        IQo, IQr, plan, att_ref_dB_per_cm_MHz=rp.att_ref
    )  # [B, plan.H, plan.W]
    return out_eff


def write_native_mat_2d(out2d: torch.Tensor, fname: str, var: str = "atten"):
    """
    out2d: [H,W] float32
    Saves a 2D matrix (no singleton 3rd dim).
    """
    arr = out2d.detach().cpu().numpy()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    scipy.io.savemat(fname + ".mat", {var: arr})


def _resize_batched(outB: torch.Tensor, target_hw: tuple[int,int]) -> torch.Tensor:
    """
    outB: [B, H, W] -> resize to target_hw using bicubic (no zeros).
    """
    B, H, W = outB.shape
    if (H, W) == target_hw:
        return outB
    return F.interpolate(outB.unsqueeze(1), size=target_hw, mode="bicubic", align_corners=False).squeeze(1)
# ===================== Pipelines =====================


def run_pipeline_mat(cfg: dict):
    rf2iq = RF2IQParams(
        Fs=cfg["Fs"], Fc=cfg.get("Fc"), Bpct=cfg.get("Bpct"),
        decim=int(cfg.get("decim", 1)), axial_dim=int(cfg.get("axial_dim", 0)),
        taps=int(cfg.get("taps", 129)),
    )
    rp    = _rp_from_dict(cfg)

    input_mode = cfg.get("input_mode", "stack").lower()  # "stack" | "single"
    input_kind = cfg.get("input_kind", "RF").upper()     # "RF" | "IQ"

    if input_mode == "single":
        # Load one 2D frame + ref -> [H,W,1]
        RFo, RFr = load_single_mat_with_ref(
            cfg["single_obj_file"], cfg["single_ref_file"],
            var_obj=cfg.get("single_var_obj", cfg.get("var_obj","RF_obj")),
            var_ref=cfg.get("single_var_ref", cfg.get("var_ref","RF_ref")),
            crop_rows_matlab=cfg.get("crop_rows_matlab"),
            crop_cols_matlab=cfg.get("crop_cols_matlab"),
            device=rp.device
        )
    else:
        # original multi-angle stack
        RFo, RFr = load_mat_angles_stack_with_ref(
            cfg["obj_dir"], cfg["ref_file"],
            var_obj=cfg.get("var_obj","RF_obj"),
            var_ref=cfg.get("var_ref","RF_ref"),
            pattern=cfg.get("pattern","angle_{:03d}.mat"),
            angles=cfg.get("angles", list(range(1,61))),
            crop_rows_matlab=cfg.get("crop_rows_matlab"),
            crop_cols_matlab=cfg.get("crop_cols_matlab"),
            device=rp.device
        )

    # shapes BEFORE RF->IQ decimation (post-crop)
    H_crop, W_crop, Na = RFo.shape

    # ===== 2) RF->IQ (or pass-through if already IQ) =====
    t1 = time.perf_counter()
    if input_kind == "RF":
        IQo1 = torch.from_numpy(
            rf2iq_downmix_lpf_butter_cpu(RFo.cpu().numpy(), Fs=rf2iq.Fs, Fc=rf2iq.Fc, axial_axis=0)
        ).to(RFo.device)
        IQr1 = torch.from_numpy(
            rf2iq_downmix_lpf_butter_cpu(RFr.cpu().numpy(), Fs=rf2iq.Fs, Fc=rf2iq.Fc, axial_axis=0)
        ).to(RFo.device)
    elif input_kind == "IQ":
        # sanity: ensure complex
        IQo1 = RFo
        IQr1 = RFr
        if not np.iscomplexobj(IQo1.cpu().numpy()):
            print("[WARN] input_kind='IQ' but object is real; casting to complex with 0j.")
            IQo1 = IQo1.to(torch.complex64) + 0j
        if not np.iscomplexobj(IQr1.cpu().numpy()):
            print("[WARN] input_kind='IQ' but reference is real; casting to complex with 0j.")
            IQr1 = IQr1.to(torch.complex64) + 0j
    else:
        raise ValueError("input_kind must be 'RF' or 'IQ'")

    # Optional decimation (only meaningful for RF→IQ case)
    if rf2iq.decim > 1:
        IQo = decimate_axial_every_M(IQo1, rf2iq.decim, axial_dim=rf2iq.axial_dim)
        IQr = decimate_axial_every_M(IQr1, rf2iq.decim, axial_dim=rf2iq.axial_dim)
    else:
        IQo, IQr = IQo1, IQr1

    t0 = time.perf_counter()
    # ===== 3) SLD (no padding) =====
    out_eff = run_sld_on_stack_nopad(IQo, IQr, rp)  # [Na, H_eff, W_eff]
    t2 = time.perf_counter()
    print(f"processing time including data conversion elapsed {t2 - t1:.2f} s")
    print(f"processing time excluding data conversion elapsed {t2 - t0:.2f} s")

    # ===== 4) sizing back to input (if requested) =====
    H_iq, W_iq = IQo.shape[0], IQo.shape[1]  # IQ tensor size you processed

    if cfg.get("resize_to_iq", False):
        out_final = _resize_batched(out_eff, (H_iq, W_iq))  # out_eff is [B,m,n]
    else:
        out_final = out_eff

    # ===== 5) write =====
    out_dir = cfg["out_dir"]; os.makedirs(out_dir, exist_ok=True)
    if input_mode == "single":
        # choose 2D or [H,W,1]
        if cfg.get("single_write_2d", True):
            write_native_mat_2d(out_final[0], os.path.join(out_dir, cfg.get("single_out_name","atten_single")), var="atten")
        else:
            write_native_mat(out_final[0:1], os.path.join(out_dir, cfg.get("single_out_name","atten_single")), var="atten")
    else:
        write_native_mat(out_final, os.path.join(out_dir, "atten_all_angles"), var="atten")

# ===================== Main =====================

if __name__ == "__main__":
    run_pipeline_mat(MAT_CONFIG)

