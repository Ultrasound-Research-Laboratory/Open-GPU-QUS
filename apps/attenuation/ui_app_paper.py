# ui_app_paper.py

import os, sys, json, time, traceback
from pathlib import Path
import numpy as np
import scipy.signal
import scipy.io

from PySide6.QtWidgets import QScrollArea
from PySide6.QtCore import QThread, Signal, Qt, QSize, QPointF
from PySide6.QtWidgets import QSizePolicy, QGridLayout, QWidget
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtGui import QPainter, QPen, QColor, QFont, QPolygonF
from PySide6.QtWidgets import (
    QApplication, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QComboBox, QTextEdit, QCheckBox, QSpinBox,
    QMessageBox, QGroupBox, QDoubleSpinBox
)

# ==== import your pipeline bits====
from GPU_SLD import (
    run_pipeline_mat,
    MAT_CONFIG,
    # lower-level pieces for manual load/process path
    load_single_mat_with_ref,
    load_mat_angles_stack_with_ref,
    rf2iq_downmix_lpf_butter_cpu,
    decimate_axial_every_M,
    write_native_mat,
    write_native_mat_2d,
    run_sld_on_stack_nopad,
    _rp_from_dict,
    # helpers reused for generic loader
    _apply_optional_crop_2d,
    _resolve_mat_var,
)
import torch
import torch.nn.functional as F


APP_TITLE = "SLD Attenuation – UIUC URL"
EPS = 1e-12


def mag_to_db_u8(arr2d: np.ndarray, dr_db: float = 60.0):
    """
    Map 2D real/complex array to 8-bit grayscale using dB scaling.
    dB = 20*log10(|x|/ref), clipped to [-dr_db, 0] then mapped to [0,255].
    Returns: gray_u8, db_min, db_max, ref_value
    """
    a = np.asarray(arr2d)

    # Enforce 2D
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.ndim != 2:
        raise ValueError(f"Preview expects 2D; got shape {a.shape}")

    # Magnitude -> float32
    if np.iscomplexobj(a):
        mag = np.abs(a.astype(np.complex64, copy=False))
    else:
        # For raw RF, compute envelope for a more informative preview
        a = scipy.signal.hilbert(a, axis=0)
        mag = np.abs(a.astype(np.complex64, copy=False)).astype(np.float32, copy=False)

    # Clean bad values
    mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)

    # Reference
    finite = mag[np.isfinite(mag)]
    if finite.size == 0:
        finite = np.array([0.0], dtype=np.float32)
    ref = float(finite.max())
    ref = max(ref, EPS)

    db = 20.0 * np.log10(np.maximum(mag, EPS) / ref)

    # Clip and map to 8-bit
    db = np.clip(db, -float(dr_db), 0.0)
    gray = ((db + dr_db) / dr_db * 255.0).astype(np.uint8)
    gray = np.ascontiguousarray(gray)

    db_min = float(np.nanmin(db)) if db.size else 0.0
    db_max = float(np.nanmax(db)) if db.size else 0.0
    return gray, db_min, db_max, ref


def qpix_from_gray(gray_u8: np.ndarray) -> QPixmap:
    """Create a QPixmap from a 2D uint8 grayscale array (contiguous)."""
    h, w = gray_u8.shape
    qimg = QImage(gray_u8.data, w, h, w, QImage.Format_Grayscale8)
    return QPixmap.fromImage(qimg.copy())


# ---------------- Helper: generic multi-MAT loader---------------
def load_mat_stack_any_pattern(
    obj_dir: str,
    ref_file: str,
    *,
    var_obj: str = "rf",
    var_ref: str = "rf2",
    crop_rows_matlab=None,
    crop_cols_matlab=None,
    device: str = "cpu",
):
    """
    Load ALL .mat files in obj_dir (any filename), excluding the reference file.
    Stacks them into RFo [H,W,B], tiles reference to [H,W,B].
    """
    obj_dir = os.path.abspath(obj_dir)
    ref_file_abs = os.path.abspath(ref_file)

    if not os.path.isdir(obj_dir):
        raise RuntimeError(f"Object folder '{obj_dir}' does not exist.")

    # collect all .mat files in folder
    all_mats = sorted(
        os.path.join(obj_dir, f)
        for f in os.listdir(obj_dir)
        if f.lower().endswith(".mat")
    )

    # exclude the reference file if it happens to live in the same folder
    mat_files = [f for f in all_mats if os.path.abspath(f) != ref_file_abs]

    if not mat_files:
        raise RuntimeError(f"No object .mat files found in '{obj_dir}' (excluding reference).")

    RFo_list = []
    H = W = None

    # Load each object mat file
    for path in mat_files:
        mobj = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
        RFo2d = _resolve_mat_var(mobj, var_obj)  # may support struct path

        if RFo2d.ndim != 2:
            raise ValueError(f"[{os.path.basename(path)}] object must be 2D; got {RFo2d.shape}")

        RFo2d = _apply_optional_crop_2d(RFo2d, crop_rows_matlab, crop_cols_matlab)

        if H is None:
            H, W = RFo2d.shape
        else:
            if RFo2d.shape != (H, W):
                raise ValueError(
                    f"Inconsistent OBJECT shape in '{os.path.basename(path)}': "
                    f"{RFo2d.shape} vs expected {(H, W)}"
                )

        RFo_list.append(RFo2d)

    RFo_np = np.stack(RFo_list, axis=2)  # [H,W,B]
    RFo_t = torch.from_numpy(RFo_np).to(device)

    # Load reference once
    mref = scipy.io.loadmat(ref_file, struct_as_record=False, squeeze_me=True)
    RFr2d = _resolve_mat_var(mref, var_ref)
    if RFr2d.ndim != 2:
        raise ValueError(f"reference var '{var_ref}' must be 2D; got {RFr2d.shape}")

    RFr2d = _apply_optional_crop_2d(RFr2d, crop_rows_matlab, crop_cols_matlab)

    if RFr2d.shape != (H, W):
        raise ValueError(
            f"reference cropped shape {RFr2d.shape} != object shape {(H, W)}"
        )

    RFr_np = np.repeat(RFr2d[:, :, None], RFo_np.shape[2], axis=2)
    RFr_t = torch.from_numpy(RFr_np).to(device)  # [H,W,B]

    return RFo_t, RFr_t


# ---------------- Worker threads (MAT only) ----------------
class Loader(QThread):
    log = Signal(str)
    loaded = Signal(object, object, dict)  # IQo(np or torch), IQr(np or torch), meta dict
    failed = Signal(str)

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

    def run(self):
        try:
            t0 = time.perf_counter()
            # Decide single vs stack
            input_mode = self.cfg.get("input_mode", "stack").lower()
            input_kind = self.cfg.get("input_kind", "RF").upper()

            if input_mode == "single":
                RFo_t, RFr_t = load_single_mat_with_ref(
                    self.cfg["single_obj_file"], self.cfg["single_ref_file"],
                    var_obj=self.cfg.get("single_var_obj", self.cfg.get("var_obj", "rf")),
                    var_ref=self.cfg.get("single_var_ref", self.cfg.get("var_ref", "rf2")),
                    crop_rows_matlab=self.cfg.get("crop_rows_matlab"),
                    crop_cols_matlab=self.cfg.get("crop_cols_matlab"),
                    device="cpu",
                )
            else:
                # NEW: generic stack loader – any *.mat filenames in obj_dir
                RFo_t, RFr_t = load_mat_stack_any_pattern(
                    self.cfg["obj_dir"], self.cfg["ref_file"],
                    var_obj=self.cfg.get("var_obj", "rf"),
                    var_ref=self.cfg.get("var_ref", "rf2"),
                    crop_rows_matlab=self.cfg.get("crop_rows_matlab"),
                    crop_cols_matlab=self.cfg.get("crop_cols_matlab"),
                    device="cpu",
                )

            # To numpy for preview & spectrum
            RFo = RFo_t.cpu().numpy()
            RFr = RFr_t.cpu().numpy()

            # Quick preview frame determination (no-op, just shape check)
            if input_kind == "RF":
                _ = RFo[:, :, 0] if RFo.ndim == 3 else RFo
            else:
                if not np.iscomplexobj(RFo):
                    RFo = RFo.astype(np.complex64) + 0j
                _ = RFo[:, :, 0] if RFo.ndim == 3 else RFo

            meta = dict(kind=input_kind, mode=input_mode, shape=RFo.shape)
            dt = time.perf_counter() - t0
            self.log.emit(f"[LOAD] Done in {dt:.2f}s. Shape={RFo.shape}, kind={input_kind}, mode={input_mode}")
            self.loaded.emit(RFo, RFr, meta)

        except Exception as e:
            tb = "".join(traceback.format_exception(e))
            self.failed.emit(tb)


class Processor(QThread):
    log = Signal(str)
    done = Signal()
    failed = Signal(str)

    def __init__(self, cfg: dict, cache: dict):
        super().__init__()
        self.cfg = cfg
        self.cache = cache  # contains IQo/IQr (numpy/torch), meta

    def run(self):
        try:
            device = self.cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            dtype_name = self.cfg.get("dtype", "float32")
            _ = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}.get(dtype_name, torch.float32)

            # Build run params (with UI overrides)
            rp = _rp_from_dict({
                **self.cfg,
                "wl_m": self.cfg["wl_m"],
                "blocksize_wl": self.cfg["blocksize_wl"],
                "overlap_pc": self.cfg["overlap_pc"],
                "fs_proc": self.cfg["fs_proc"],
                "freq_L_MHz": self.cfg["freq_L_MHz"],
                "freq_H_MHz": self.cfg["freq_H_MHz"],
                "passband_shift_MHz": self.cfg["passband_shift_MHz"],
                "att_ref": self.cfg["att_ref"],
                "batch_slices": self.cfg.get("batch_slices", 64),
                "device": device,
                "dtype": dtype_name,
            })

            input_kind = self.cache["meta"]["kind"].upper()
            # Prepare IQ (on CPU first, then move)
            if input_kind == "RF":
                RFo = self.cache["RFo"]  # numpy [H,W,B] or [H,W]
                RFr = self.cache["RFr"]
                IQo_np = rf2iq_downmix_lpf_butter_cpu(RFo, Fs=self.cfg["Fs"], Fc=self.cfg["Fc"], axial_axis=self.cfg.get("axial_dim", 0))
                IQr_np = rf2iq_downmix_lpf_butter_cpu(RFr, Fs=self.cfg["Fs"], Fc=self.cfg["Fc"], axial_axis=self.cfg.get("axial_dim", 0))
            else:
                IQo_np = self.cache["RFo"]
                IQr_np = self.cache["RFr"]
                if not np.iscomplexobj(IQo_np):
                    IQo_np = IQo_np.astype(np.complex64) + 0j
                if not np.iscomplexobj(IQr_np):
                    IQr_np = IQr_np.astype(np.complex64) + 0j

            # Optional decimation
            decim = int(self.cfg.get("decim", 1))
            IQo = torch.from_numpy(IQo_np)
            IQr = torch.from_numpy(IQr_np)
            if decim > 1:
                IQo = decimate_axial_every_M(IQo, decim, axial_dim=self.cfg.get("axial_dim", 0))
                IQr = decimate_axial_every_M(IQr, decim, axial_dim=self.cfg.get("axial_dim", 0))

            # Move to device
            IQo = IQo.to(device)
            IQr = IQr.to(device)

            # Start of the processing time (after conversion from RF into IQ)
            t0 = time.perf_counter()
            # Run SLD
            outB = run_sld_on_stack_nopad(IQo, IQr, rp)  # [B,H_eff,W_eff]

            if self.cfg.get("resize_to_iq", False):
                # IQo is the actual IQ you processed (already decimated if RF)
                H_iq, W_iq = IQo.shape[0], IQo.shape[1]
                outB = F.interpolate(
                    outB.unsqueeze(1),
                    size=(H_iq, W_iq),
                    mode="bicubic",
                    align_corners=False
                ).squeeze(1)

            # Write outputs
            out_dir = self.cfg["out_dir"]; os.makedirs(out_dir, exist_ok=True)
            if self.cfg.get("input_mode", "stack").lower() == "single":
                if self.cfg.get("single_write_2d", True):
                    write_native_mat_2d(outB[0], os.path.join(out_dir, self.cfg.get("single_out_name", "atten_single")), var="atten")
                else:
                    write_native_mat(outB[0:1], os.path.join(out_dir, self.cfg.get("single_out_name", "atten_single")), var="atten")
            else:
                write_native_mat(outB, os.path.join(out_dir, "atten_all_angles"), var="atten")

            dt = time.perf_counter() - t0
            self.log.emit(f"[PROCESS] Done in {dt:.2f}s (processing & Output).")
            self.done.emit()

        except Exception as e:
            tb = "".join(traceback.format_exception(e))
            self.failed.emit(tb)


# -------------------- Main Window (MAT only) --------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1220, 760)

        self.cache = {}      # holds loaded arrays & meta
        self._last_spectrum = None  # (freqs_mhz, mag_db, is_iq)
        # ---------- Illinois / BRL banner ----------
        self.banner = QLabel(
            "University of Illinois Urbana-Champaign\n"
            "Ultrasound Research Lab – SLD Attenuation Tool"
        )
        self.banner.setAlignment(Qt.AlignCenter)
        self.banner.setStyleSheet(
            "background-color: #13294B;"
            "color: white;"
            "font-weight: bold;"
            "padding: 10px;"
            "font-size: 16px;"
            "border-bottom: 3px solid #E84A27;"
        )
        # --- MAT group ---
        mat_box = QGroupBox("MAT mode")
        self.mat_obj_dir = self._path_row("Object folder (*.mat)…", is_file=False)
        self.mat_ref_file = self._path_row("Reference file (reference.mat)…", is_file=True)
        self.mat_out_dir  = self._path_row("Output folder…", is_file=False)

        self.mat_input_kind = QComboBox(); self.mat_input_kind.addItems(["RF", "IQ"])
        self.mat_single = QCheckBox("Single 2D frame mode")
        self.mat_single_obj = self._path_row("Single object .mat…", is_file=True)
        self.mat_single_ref = self._path_row("Single reference .mat…", is_file=True)

        # SLD knobs (shared, editable)
        self.wl_m = self._dspin("Wavelength (m):", 0.00005, 0.005, default=MAT_CONFIG.get("wl_m", 0.00038), step=0.00001, decimals=6)
        self.blocksize = self._spin("Block size (λ):", 4, 256, default=MAT_CONFIG.get("blocksize_wl", 40))
        self.overlap = self._dspin("Overlap (0–0.99):", 0.0, 0.99, default=MAT_CONFIG.get("overlap_pc", 0.875), step=0.005, decimals=3)
        self.fs_proc = self._dspin(
            "fs_proc (MHz):", 0.01, 200.0,
            default=MAT_CONFIG.get("fs_proc", 6.392543818714412e06) / 1e6,
            step=0.01, decimals=3
        )
        self.freqL = self._dspin("Freq L (MHz):", -20.0, 20.0, default=MAT_CONFIG.get("freq_L_MHz", -1.1), step=0.05, decimals=3)
        self.freqH = self._dspin("Freq H (MHz):", -20.0, 20.0, default=MAT_CONFIG.get("freq_H_MHz",  1.1), step=0.05, decimals=3)
        self.fshift = self._dspin("Passband shift (MHz):", 0.0, 20.0, default=MAT_CONFIG.get("passband_shift_MHz", 3.6), step=0.1, decimals=2)
        self.attref = self._dspin("att_ref (dB/cm/MHz):", 0.0, 5.0, default=MAT_CONFIG.get("att_ref", 0.2), step=0.05, decimals=3)
        self.mat_batch = self._spin("Batch slices:", 1, 512, default=MAT_CONFIG.get("batch_slices", 60))

        self.mat_resize_to_iq = QCheckBox("Resize output to RF→IQ size (after RF→IQ decim)")
        self.mat_resize_to_iq.setChecked(bool(MAT_CONFIG.get("resize_to_iq", False)))
        self.dz_mm = self._dspin("dz (mm/pixel):", 0.001, 5.0, default=0.01505, step=0.001, decimals=4)
        self.dx_mm = self._dspin("dx (mm/pixel):", 0.001, 5.0, default=0.4173, step=0.001, decimals=4)

        mat_grid = QGridLayout()
        widgets = [
            self.wl_m["box"], self.blocksize["box"], self.overlap["box"], self.fs_proc["box"],
            self.freqL["box"], self.freqH["box"], self.fshift["box"], self.attref["box"],
            self.mat_batch["box"], self.dz_mm["box"], self.dx_mm["box"]
        ]
        for i, w in enumerate(widgets):
            mat_grid.addWidget(w, i // 6, i % 6)

        mat_layout = QVBoxLayout()
        mat_layout.addLayout(self.mat_obj_dir["layout"])
        mat_layout.addLayout(self.mat_ref_file["layout"])
        mat_layout.addLayout(self.mat_out_dir["layout"])
        mat_layout.addLayout(mat_grid)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Input kind:")); row1.addWidget(self.mat_input_kind)
        row1.addSpacing(20); row1.addWidget(self.mat_single)
        mat_layout.addLayout(row1)
        mat_layout.addLayout(self.mat_single_obj["layout"])
        mat_layout.addLayout(self.mat_single_ref["layout"])

        row2 = QHBoxLayout()
        row2.addWidget(self.mat_resize_to_iq)
        row2.addStretch(1)
        mat_layout.addLayout(row2)
        mat_box.setLayout(mat_layout)

        # --- Common controls (top bar) ---
        self.device = QComboBox(); self.device.addItems(["cuda", "cpu"])
        self.device.setCurrentText("cuda" if "cuda" in MAT_CONFIG.get("device", "cuda") and torch.cuda.is_available() else "cpu")
        self.dtype = QComboBox(); self.dtype.addItems(["float32", "float16", "bfloat16"])
        self.dtype.setCurrentText(MAT_CONFIG.get("dtype", "float32"))

        self.load_btn = QPushButton("Load Data")
        self.start_btn = QPushButton("Start (Process)")
        self.open_out_btn = QPushButton("Open Output Folder"); self.open_out_btn.setEnabled(False)
        self.preview_db = QCheckBox("Preview in dB"); self.preview_db.setChecked(True)
        self.preview_dr = QDoubleSpinBox(); self.preview_dr.setRange(10.0, 120.0); self.preview_dr.setDecimals(1); self.preview_dr.setSingleStep(5.0); self.preview_dr.setValue(60.0)

        top = QHBoxLayout()
        top.addWidget(QLabel("Device:")); top.addWidget(self.device)
        top.addWidget(QLabel("dtype:"));  top.addWidget(self.dtype); top.addStretch(1)
        top.addWidget(self.load_btn); top.addWidget(self.start_btn); top.addWidget(self.open_out_btn)
        top.addSpacing(16)
        top.addWidget(self.preview_db); top.addWidget(QLabel("DR (dB):")); top.addWidget(self.preview_dr)

        # Preview (image) – shown only in single-2D MAT mode
        self.preview = QLabel("Preview (single 2D only)")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(QSize(320, 240))
        self.preview.setStyleSheet("border:1px solid #bbb; background:#fafafa;")
        self.preview.setVisible(False)  # default hidden
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.dx_mm["spin"].valueChanged.connect(self._refresh_preview)
        self.dz_mm["spin"].valueChanged.connect(self._refresh_preview)
        self.preview_db.stateChanged.connect(self._refresh_preview)
        self.preview_dr.valueChanged.connect(self._refresh_preview)

        # Spectrum – always shown (after load)
        self.spectrum = QLabel("FFT Spectrum: (load first)")
        self.spectrum.setAlignment(Qt.AlignCenter)
        self.spectrum.setMinimumSize(QSize(520, 260))
        self.spectrum.setStyleSheet("border:1px solid #bbb; background:#fafafa;")

        # Log
        self.log = QTextEdit(); self.log.setReadOnly(True)

        # Layout root
        root = QVBoxLayout()
        root.addWidget(self.banner)
        root.addLayout(top)
        # Put the MAT panel + visuals into a scrollable container
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.addWidget(mat_box)

        # visuals row
        self.visuals_row = QHBoxLayout()
        self.visuals_row.setSpacing(8)
        self.visuals_row.addWidget(self.preview, 1)
        self.visuals_row.addWidget(self.spectrum, 1)
        scroll_layout.addLayout(self.visuals_row)

        scroll.setWidget(scroll_content)

        root.addWidget(scroll, stretch=3)  # scrollable upper content
        root.addWidget(self.log, stretch=1)  # log always visible
        self.setLayout(root)

        # Wire signals
        self.mat_single.stateChanged.connect(self._update_preview_visibility)
        self.load_btn.clicked.connect(self._load)
        self.start_btn.clicked.connect(self._start)
        self.open_out_btn.clicked.connect(self._open_output)

        # Auto update red band when L/H or fshift change
        self.freqL["spin"].valueChanged.connect(self._update_spectrum_overlay_from_ui)
        self.freqH["spin"].valueChanged.connect(self._update_spectrum_overlay_from_ui)
        self.fshift["spin"].valueChanged.connect(self._update_spectrum_overlay_from_ui)

        # Set initial visibility and defaults
        self._update_preview_visibility()
        self._load_defaults()

        self.loader = None
        self.proc = None

    # ---- small UI helpers ----
    def _update_preview_visibility(self):
        """Preview image is shown only for single-2D mode."""
        self.preview.setVisible(self.mat_single.isChecked())

    def _append(self, msg: str):
        self.log.append(msg)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _path_row(self, label, is_file: bool):
        le = QLineEdit()
        btn = QPushButton("Browse…")
        hl = QHBoxLayout()
        hl.addWidget(QLabel(label)); hl.addWidget(le); hl.addWidget(btn)

        def browse():
            if is_file:
                path, _ = QFileDialog.getOpenFileName(self, "Select file")
            else:
                path = QFileDialog.getExistingDirectory(self, "Select folder")
            if path:
                le.setText(path)

        btn.clicked.connect(browse)
        return {"layout": hl, "edit": le}

    def _spin(self, title: str, lo: int, hi: int, default: int):
        box = QGroupBox(title); sb = QSpinBox(); sb.setRange(lo, hi); sb.setValue(default)
        lay = QHBoxLayout(); lay.addWidget(sb); box.setLayout(lay)
        return {"box": box, "spin": sb}

    def _dspin(self, title: str, lo: float, hi: float, default: float, step: float, decimals: int):
        box = QGroupBox(title); ds = QDoubleSpinBox(); ds.setRange(lo, hi); ds.setDecimals(decimals)
        ds.setSingleStep(step); ds.setValue(float(default))
        lay = QHBoxLayout(); lay.addWidget(ds); box.setLayout(lay)
        return {"box": box, "spin": ds}

    def _nice_step(self, span, target_ticks=6):
        span = float(max(span, 1e-12))
        raw = span / target_ticks
        base = 10 ** np.floor(np.log10(raw))
        for m in (1, 2, 2.5, 5, 10):
            if raw <= m * base:
                return m * base
        return 10 * base

    def _imagesc_pixmap(self, gray_u8: np.ndarray, dx_mm: float, dz_mm: float,
                        canvas_w: int, canvas_h: int) -> QPixmap:
        """
        Render a grayscale image with MATLAB-like imagesc(x,z,...) + axis image:
        - axes in centimeters (x lateral, z axial)
        - equal data-unit scaling (1 cm in x = 1 cm in z)
        - grayscale
        """
        H, W = gray_u8.shape

        # physical extents in cm (dx,dz are mm/pixel in the UI)
        x_max_cm = (W * dx_mm) / 10.0
        z_max_cm = (H * dz_mm) / 10.0
        x_max_cm = float(max(x_max_cm, 1e-9))
        z_max_cm = float(max(z_max_cm, 1e-9))

        # canvas
        L, R, T, B = 60, 22, 20, 38  # margins
        Wp = max(canvas_w - L - R, 50)
        Hp = max(canvas_h - T - B, 50)

        # axis image: plot rect aspect must equal (x_max / z_max)
        desired_aspect = x_max_cm / z_max_cm  # width/height in data units
        if (Wp / Hp) > desired_aspect:
            plot_h = Hp
            plot_w = int(round(plot_h * desired_aspect))
        else:
            plot_w = Wp
            plot_h = int(round(plot_w / desired_aspect))

        # center plot rect within axes box
        ox = L + (Wp - plot_w) // 2
        oy = T + (Hp - plot_h) // 2

        # image canvas
        img = QImage(canvas_w, canvas_h, QImage.Format_RGB32)
        img.fill(QColor(250, 250, 250))
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # draw axes frame
        p.setPen(QPen(QColor(200, 200, 200), 1))
        p.drawRect(L, T, Wp, Hp)

        # draw grayscale image scaled into plot rect (preserves axis-image scaling)
        qimg_gray = QImage(gray_u8.data, W, H, W, QImage.Format_Grayscale8).copy()
        p.drawImage(ox, oy, qimg_gray.scaled(plot_w, plot_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation))

        # ticks & grid (in cm)
        sx = self._nice_step(x_max_cm, 6)
        sz = self._nice_step(z_max_cm, 6)
        xticks = np.arange(0.0, x_max_cm + 0.5 * sx, sx)
        zticks = np.arange(0.0, z_max_cm + 0.5 * sz, sz)

        # tick labels & axis labels
        p.setPen(QPen(QColor(0, 0, 0), 1))
        font = QFont(); font.setPointSize(9); p.setFont(font)

        for xv in xticks:
            X = ox + int(round((xv / x_max_cm) * plot_w))
            p.drawText(X - 14, T + Hp + 16, f"{xv:.2g}")
        p.drawText(L + Wp - 42, T + Hp + 32, "x (cm)")

        for zv in zticks:
            Y = oy + int(round((zv / z_max_cm) * plot_h))
            ytxt = min(max(Y + 4, T + 12), T + Hp - 4)
            p.drawText(6, ytxt, f"{zv:.2g}")
        p.drawText(8, T - 10, "z (cm)")

        p.end()
        return QPixmap.fromImage(img)

    def _compute_spectrum_1d(self, obj_np: np.ndarray, meta: dict, cfg: dict):
        """
        Build a 1D axial signal (mean over lateral/angles), apply Hann window,
        FFT, return (freqs_MHz, mag_dB_norm, is_iq), normalized to 0 dB peak.
        """
        # collapse to 1D along axial (axis 0)
        if obj_np.ndim == 2:
            sig = obj_np.mean(axis=1)
        elif obj_np.ndim == 3:
            sig = obj_np.mean(axis=(1, 2))
        else:
            raise ValueError(f"Unexpected shape for spectrum: {obj_np.shape}")

        # Choose sampling rate (IQ uses processed fs; RF uses raw Fs)
        is_iq = (meta.get("kind", "RF").upper() == "IQ")
        Fs = float(cfg.get("fs_proc")) if is_iq else float(cfg.get("Fs"))
        Fs = max(Fs, 1.0)

        # Hann window to reduce leakage
        H = sig.shape[0]
        win = np.hanning(H)
        sig_win = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0) * win

        # FFT
        nfft = 1 << int(np.ceil(np.log2(max(H, 16))))
        if is_iq:
            S = np.fft.fft(sig_win, n=nfft)
            S = np.fft.fftshift(S)
            freqs = np.fft.fftfreq(nfft, d=1.0 / Fs)
            freqs = np.fft.fftshift(freqs)
        else:
            S = np.fft.rfft(sig_win, n=nfft)
            freqs = np.fft.rfftfreq(nfft, d=1.0 / Fs)

        mag = np.abs(S).astype(np.float64)
        mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
        ref = np.max(mag) if mag.size else 1.0
        mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12) / max(ref, 1e-12))

        return (freqs / 1e6), mag_db, is_iq  # MHz, dB

    def _draw_spectrum_pixmap(self, freqs_mhz: np.ndarray, mag_db: np.ndarray,
                              *, width=820, height=260, dr_db=80.0,
                              band_L_MHz=None, band_H_MHz=None,
                              hline_dbs=None):
        img = QImage(width, height, QImage.Format_RGB32)
        img.fill(QColor(250, 250, 250))
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing, True)

        # plot rect with margins
        L, R, T, B = 60, 20, 20, 36
        Wp = width - L - R
        Hp = height - T - B
        p.setPen(QPen(QColor(200, 200, 200), 1))
        p.drawRect(L, T, Wp, Hp)

        # clip mag to [-dr, 0]
        y = np.clip(mag_db, -float(dr_db), 0.0)

        # x range
        x0, x1 = float(freqs_mhz.min()), float(freqs_mhz.max())
        if x0 == x1:
            x0, x1 = -1.0, 1.0

        # ticks
        span = x1 - x0
        step = 10 ** np.floor(np.log10(span / 6.0))
        for mult in [1, 2, 2.5, 5, 10]:
            if span / (step * mult) <= 6.0:
                step *= mult
                break
        xticks = np.arange(np.ceil(x0 / step) * step, x1 + step * 0.5, step)
        yticks = [-dr_db, -dr_db / 2.0, 0.0]

        # grid
        p.setPen(QPen(QColor(220, 220, 220), 1))
        for xv in xticks:
            X = L + int(round((xv - x0) / (x1 - x0) * Wp))
            p.drawLine(X, T, X, T + Hp)
        for yv in yticks:
            Y = T + int(round((0 - yv) / dr_db * Hp))
            p.drawLine(L, Y, L + Wp, Y)

        # axes labels
        p.setPen(QPen(QColor(0, 0, 0), 1))
        font = QFont(); font.setPointSize(9); p.setFont(font)
        for xv in xticks:
            X = L + int(round((xv - x0) / (x1 - x0) * Wp))
            p.drawText(X - 18, T + Hp + 16, f"{xv:.2g}")
        p.drawText(L + Wp - 50, T + Hp + 30, "MHz")
        for yv in yticks:
            Y = T + int(round((0 - yv) / dr_db * Hp))
            p.drawText(4, Y + 4, f"{yv:.0f} dB")

        # spectrum line (downsample if needed)
        N = len(freqs_mhz)
        stride = max(1, N // max(Wp, 1))
        Xs = L + ((freqs_mhz[::stride] - x0) / (x1 - x0) * Wp)
        Ys = T + ((0 - y[::stride]) / dr_db * Hp)
        poly = QPolygonF([QPointF(float(x), float(y_)) for x, y_ in zip(Xs, Ys)])

        p.setPen(QPen(QColor(30, 60, 160), 2))
        p.drawPolyline(poly)

        # reference horizontal lines (clamped labels)
        if hline_dbs:
            p.setPen(QPen(QColor(120, 120, 120), 1, Qt.DashLine))
            fm = p.fontMetrics()
            text_h = fm.height()
            for lvl in hline_dbs:
                if not (-dr_db <= lvl <= 0.0):
                    continue
                Y = T + int(round((0 - lvl) / dr_db * Hp))
                p.drawLine(L, Y, L + Wp, Y)
                y_text = max(T + text_h, min(Y - 3, T + Hp - 2))
                p.drawText(L + 6, y_text, f"{lvl:.0f} dB")

        # current band overlay
        if (band_L_MHz is not None) and (band_H_MHz is not None):
            p.setPen(QPen(QColor(180, 50, 50), 1, Qt.DashLine))
            for xv in [band_L_MHz, band_H_MHz]:
                if xv < x0 or xv > x1:
                    continue
                X = L + int(round((xv - x0) / (x1 - x0) * Wp))
                p.drawLine(X, T, X, T + Hp)

        p.end()
        return QPixmap.fromImage(img)

    def _load_defaults(self):
        # MAT
        self.mat_obj_dir["edit"].setText(MAT_CONFIG.get("obj_dir", ""))
        self.mat_ref_file["edit"].setText(MAT_CONFIG.get("ref_file", ""))
        self.mat_out_dir["edit"].setText(MAT_CONFIG.get("out_dir", ""))
        if "fs_proc" in MAT_CONFIG:
            self.fs_proc["spin"].setValue(float(MAT_CONFIG["fs_proc"]) / 1e6)
        self.mat_input_kind.setCurrentText(MAT_CONFIG.get("input_kind", "RF").upper())
        self.mat_single.setChecked(MAT_CONFIG.get("input_mode", "stack").lower() == "single")
        self.mat_single_obj["edit"].setText(MAT_CONFIG.get("single_obj_file", ""))
        self.mat_single_ref["edit"].setText(MAT_CONFIG.get("single_ref_file", ""))

    def _refresh_preview(self):
        """Rebuild preview gray and render with axes using current dx/dz."""
        if not self.cache:
            return
        meta = self.cache.get("meta", {})
        if meta.get("mode", "stack").lower() != "single":
            return

        IQo = self.cache["RFo"]
        img = IQo[:, :, 0] if IQo.ndim == 3 else IQo

        # make 8-bit grayscale (dB or linear)
        if self.preview_db.isChecked():
            gray, _, _, _ = mag_to_db_u8(img, dr_db=self.preview_dr.value())
        else:
            a = np.abs(img) if np.iscomplexobj(img) else img
            finite = a[np.isfinite(a)]
            lo, hi = (np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0))
            if hi <= lo:
                hi = lo + 1e-6
            b = np.clip((a - lo) / (hi - lo), 0, 1)
            gray = np.ascontiguousarray((b * 255).astype(np.uint8))

        self._preview_gray_u8 = gray
        self._last_shape = gray.shape

        # render with axes (imagesc + axis image)
        dx = float(self.dx_mm["spin"].value())
        dz = float(self.dz_mm["spin"].value())
        pix = self._imagesc_pixmap(gray, dx, dz, self.preview.width(), self.preview.height())
        self.preview.setPixmap(pix)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_preview()

    # ---- gather configs with UI overrides ----
    def _gather_mat_cfg(self) -> dict:
        cfg = dict(MAT_CONFIG)
        cfg["obj_dir"]  = self.mat_obj_dir["edit"].text()
        cfg["ref_file"] = self.mat_ref_file["edit"].text()
        cfg["out_dir"]  = self.mat_out_dir["edit"].text()
        cfg["input_kind"] = self.mat_input_kind.currentText()
        cfg["input_mode"] = "single" if self.mat_single.isChecked() else "stack"
        cfg["single_obj_file"] = self.mat_single_obj["edit"].text()
        cfg["single_ref_file"] = self.mat_single_ref["edit"].text()

        # SLD knobs
        cfg["wl_m"] = self.wl_m["spin"].value()
        cfg["blocksize_wl"] = self.blocksize["spin"].value()
        cfg["overlap_pc"] = self.overlap["spin"].value()
        cfg["fs_proc"] = self.fs_proc["spin"].value() * 1e6
        cfg["freq_L_MHz"] = self.freqL["spin"].value()
        cfg["freq_H_MHz"] = self.freqH["spin"].value()
        cfg["passband_shift_MHz"] = self.fshift["spin"].value()
        cfg["att_ref"] = self.attref["spin"].value()
        cfg["batch_slices"] = self.mat_batch["spin"].value()

        cfg["resize_to_iq"] = self.mat_resize_to_iq.isChecked()

        cfg["device"] = self.device.currentText()
        cfg["dtype"]  = self.dtype.currentText()
        return cfg

    # ---- spectrum overlay refresh ----
    def _update_spectrum_overlay_from_ui(self):
        """Overlay current UI band (Freq L/H [+shift for RF]) on the last spectrum and redraw."""
        if not self.cache:
            return  # nothing loaded yet

        # reuse last spectrum if we have it; otherwise recompute
        if self._last_spectrum:
            freqs_mhz, mag_db, is_iq = self._last_spectrum
        else:
            IQo = self.cache["RFo"]
            meta = self.cache["meta"]
            cfg_now = self._gather_mat_cfg()
            freqs_mhz, mag_db, is_iq = self._compute_spectrum_1d(IQo, meta, cfg_now)
            self._last_spectrum = (freqs_mhz, mag_db, is_iq)

        cfg_now = self._gather_mat_cfg()
        L_MHz_ui = float(cfg_now.get("freq_L_MHz", -1.0))
        H_MHz_ui = float(cfg_now.get("freq_H_MHz",  1.0))
        fshift = float(cfg_now.get("passband_shift_MHz", 0.0))

        if is_iq:
            band_L = L_MHz_ui
            band_H = H_MHz_ui
        else:
            band_L = L_MHz_ui + fshift
            band_H = H_MHz_ui + fshift

        spec_pix = self._draw_spectrum_pixmap(
            freqs_mhz, mag_db,
            width=max(self.spectrum.width(), 820),
            height=max(self.spectrum.height(), 260),
            dr_db=max(self.preview_dr.value(), 60.0),
            band_L_MHz=band_L, band_H_MHz=band_H,
            hline_dbs=[-6, -15, -20]
        )
        self.spectrum.setPixmap(spec_pix)

    # ---- buttons ----
    def _load(self):
        cfg = self._gather_mat_cfg()

        # quick validation
        try:
            if cfg["input_mode"] == "single":
                if not (os.path.isfile(cfg["single_obj_file"]) and os.path.isfile(cfg["single_ref_file"])):  # noqa
                    raise RuntimeError("Pick valid single object/ref .mat files.")
                if not os.path.isdir(cfg["out_dir"]):
                    os.makedirs(cfg["out_dir"], exist_ok=True)
            else:
                if not os.path.isdir(cfg["obj_dir"]):
                    raise RuntimeError("Invalid MAT object folder.")
                if not os.path.isfile(cfg["ref_file"]):
                    raise RuntimeError("Invalid MAT reference file.")
                if not os.path.isdir(cfg["out_dir"]):
                    os.makedirs(cfg["out_dir"], exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Invalid config", str(e))
            return

        self.log.clear()
        self.load_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.open_out_btn.setEnabled(False)

        self.loader = Loader(cfg)
        self.loader.log.connect(self._append)
        self.loader.loaded.connect(self._on_loaded)
        self.loader.failed.connect(self._on_load_failed)
        self.loader.start()

    def _on_loaded(self, IQo, IQr, meta):
        # cache
        self.cache = {"RFo": IQo, "RFr": IQr, "meta": meta}
        self._last_spectrum = None  # reset; we’ll recompute with current config

        # 1) PREVIEW image: only for single 2D
        try:
            show_preview = (meta.get("mode", "stack").lower() == "single")
            if show_preview:
                img = IQo[:, :, 0] if IQo.ndim == 3 else IQo
                if self.preview_db.isChecked():
                    gray, db_min, db_max, _ = mag_to_db_u8(img, dr_db=self.preview_dr.value())
                    pix = qpix_from_gray(gray)
                else:
                    a = img
                    if np.iscomplexobj(a):
                        a = np.abs(a)
                    finite = a[np.isfinite(a)]
                    lo, hi = (np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0))
                    if hi <= lo:
                        hi = lo + 1e-6
                    b = np.clip((a - lo) / (hi - lo), 0, 1)
                    pix = qpix_from_gray(np.ascontiguousarray((b * 255).astype(np.uint8)))
                self._current_pixmap = pix
                self._last_shape = img.shape
                self._refresh_preview()
                self.preview.setVisible(True)
            else:
                self.preview.clear()
                self.preview.setText("Preview (single 2D only)")
                self.preview.setVisible(False)
        except Exception as e:
            self._append(f"[WARN] Preview failed: {e}")
            self.preview.setText("Preview failed.")
            self.preview.setVisible(True)

        # 2) FFT SPECTRUM: always compute & show (object signal)
        try:
            cfg = self._gather_mat_cfg()
            freqs_mhz, mag_db, is_iq = self._compute_spectrum_1d(IQo, meta, cfg)
            self._last_spectrum = (freqs_mhz, mag_db, is_iq)

            # Current band overlay: RF uses [L+shift, H+shift]; IQ uses [L, H]
            L_MHz = float(cfg.get("freq_L_MHz", -1.0))
            H_MHz = float(cfg.get("freq_H_MHz", 1.0))
            fshift = float(cfg.get("passband_shift_MHz", 0.0))
            band_L, band_H = (L_MHz + fshift, H_MHz + fshift) if not is_iq else (L_MHz, H_MHz)

            spec_pix = self._draw_spectrum_pixmap(
                freqs_mhz, mag_db,
                width=max(self.spectrum.width(), 820),
                height=max(self.spectrum.height(), 260),
                dr_db=max(self.preview_dr.value(), 60.0),
                band_L_MHz=band_L, band_H_MHz=band_H,
                hline_dbs=[-6, -15, -20]
            )
            self.spectrum.setPixmap(spec_pix)
        except Exception as e:
            self._append(f"[WARN] Spectrum failed: {e}")
            self.spectrum.setText("FFT Spectrum failed.")

        self._append("[READY] Data cached. Adjust SLD params, then click Start (Process).")
        self.load_btn.setEnabled(True)
        self.start_btn.setEnabled(True)

    def _on_load_failed(self, tb: str):
        self._append("[ERROR] Load failed:\n" + tb)
        self.load_btn.setEnabled(True)
        self.start_btn.setEnabled(False)

    def _start(self):
        if not self.cache:
            QMessageBox.information(self, "No data", "Click 'Load Data' first.")
            return

        cfg = self._gather_mat_cfg()

        self._append(f"[RUN] Processing with blocksize={cfg['blocksize_wl']}, "
                     f"freq=[{cfg['freq_L_MHz']}, {cfg['freq_H_MHz']}] MHz")
        self.load_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.open_out_btn.setEnabled(False)

        self.proc = Processor(cfg, self.cache)
        self.proc.log.connect(self._append)
        self.proc.done.connect(self._on_proc_done)
        self.proc.failed.connect(self._on_proc_failed)
        self.proc.start()

    def _on_proc_done(self):
        self._append("[OK] Processing finished.")
        self.load_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        self.open_out_btn.setEnabled(True)

    def _on_proc_failed(self, tb: str):
        self._append("[ERROR] Processing failed:\n" + tb)
        self.load_btn.setEnabled(True)
        self.start_btn.setEnabled(True)

    def _open_output(self):
        path = self._gather_mat_cfg()["out_dir"]
        if os.path.isdir(path):
            try:
                os.startfile(path)
            except AttributeError:
                import subprocess
                subprocess.Popen(["xdg-open", path])


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
