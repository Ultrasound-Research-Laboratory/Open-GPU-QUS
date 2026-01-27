# ui_app_bsc.py
import os, sys, time, traceback
import numpy as np

from PySide6.QtCore import Qt, QSize, QThread, Signal, QPointF
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QPolygonF
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QTextEdit, QMessageBox, QSizePolicy, QScrollArea
)

# ---- backend already have ----
from GPU_BSC import (
    read_iq_file_int32_interleaved,
    run_qt_bsc_single_slice,
    run_qt_bsc,
)

APP_TITLE = "BSC – UIUC URL"
EPS = 1e-12

# ---- default paths (edit to your environment) ----
DEFAULT_PATHS = dict(
    ref_dir = r"X:\ml132\Calculate_BSC_for_QT_big_phantom\BigPhantom\refl",
    obj_dir = r"Y:\Sunnybrook_patients_raw_data\QT002\QT002 RAW\da\refl",
    out_dir_esd = r"X:\ml132\GPU_QUS\BSCApp_for_paper\esd",
    out_dir_eac = r"X:\ml132\GPU_QUS\BSCApp_for_paper\eac",
    BSC_ref_file = r"X:\ml132\BSC_for_Sunnybrook\BigPhantom_BSC.txt",
)

# ---------------- helpers ----------------
def mag_to_db_u8(arr2d: np.ndarray, dr_db: float = 60.0):
    a = np.asarray(arr2d)
    if a.ndim != 2:
        raise ValueError(f"Preview expects 2D; got {a.shape}")
    mag = np.abs(a.astype(np.complex64, copy=False)) if np.iscomplexobj(a) else np.abs(a.astype(np.float32, copy=False))
    mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)

    finite = mag[np.isfinite(mag)]
    ref = float(finite.max()) if finite.size else 1.0
    ref = max(ref, EPS)

    db = 20.0 * np.log10(np.maximum(mag, EPS) / ref)
    db = np.clip(db, -float(dr_db), 0.0)
    gray = ((db + dr_db) / dr_db * 255.0).astype(np.uint8)
    return np.ascontiguousarray(gray)


def spectrum_from_slice_avg(iq2d: np.ndarray, fs_hz: float, nfft: int):
    """
    Averaged spectrum (less noisy):
      - FFT along axial dimension for EACH lateral column
      - average power spectrum across columns
      - return baseband freqs (MHz) and mag in dB normalized to peak=0 dB
    """
    x = np.asarray(iq2d)
    if x.ndim != 2:
        raise ValueError("spectrum_from_slice_avg expects a 2D slice [H,W]")

    H, W = x.shape
    fs_hz = float(max(fs_hz, 1.0))
    N = int(max(16, nfft))

    win = np.hanning(H).astype(np.float64)[:, None]  # [H,1]
    xw = (x.astype(np.complex64, copy=False) * win)

    S = np.fft.fft(xw, n=N, axis=0)
    S = np.fft.fftshift(S, axes=0)

    P = (S.real.astype(np.float64)**2 + S.imag.astype(np.float64)**2).mean(axis=1)  # [N]
    f = np.fft.fftshift(np.fft.fftfreq(N, d=1.0/fs_hz))

    ref = float(np.max(P)) if P.size else 1.0
    ref = max(ref, 1e-12)
    mag_db = 10.0 * np.log10(np.maximum(P, 1e-12) / ref)  # power dB

    return (f / 1e6), mag_db  # MHz (baseband), dB


# ---------------- workers ----------------
class LoadWorker(QThread):
    log = Signal(str)
    done = Signal(object)  # payload dict
    failed = Signal(str)

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

    def run(self):
        try:
            angle = int(self.cfg["angle_lv"])
            z = int(self.cfg["z_idx"])

            # --- load OBJECT ONLY for preview ---
            obj_path = os.path.join(self.cfg["obj_dir"], f"unit1_{angle}")
            self.log.emit(f"[LOAD] Object: unit1_{angle} (z={z})")
            IQ_obj = read_iq_file_int32_interleaved(obj_path)  # [552,192,Nz]
            Nz = IQ_obj.shape[2]
            if not (0 <= z < Nz):
                raise IndexError(f"z_idx {z} out of range 0..{Nz-1}")

            iq2d_obj = IQ_obj[:, :, z]
            payload = {"iq2d_obj": iq2d_obj, "Nz": Nz}

            # --- load TXT table (passband MHz, BSC) ---
            tab_path = self.cfg["BSC_ref_file"]
            self.log.emit(f"[LOAD] BSC TXT: {tab_path}")
            tab = np.loadtxt(tab_path, delimiter=',').astype(np.float32)
            if tab.ndim != 2 or tab.shape[1] < 2:
                raise ValueError("BSC_ref_file must be 2 columns: MHz,value (CSV)")
            payload["bsc_tab"] = tab

            self.done.emit(payload)
        except Exception:
            self.failed.emit(traceback.format_exc())


class SingleWorker(QThread):
    log = Signal(str)
    done = Signal(object, object, float)  # esd2d,eac2d,t
    failed = Signal(str)

    def __init__(self, params: dict, angle: int, z: int, write: bool):
        super().__init__()
        self.params = params
        self.angle = angle
        self.z = z
        self.write = write

    def run(self):
        try:
            t0 = time.perf_counter()
            esd2d, eac2d, t = run_qt_bsc_single_slice(self.params, self.angle, self.z, write=self.write)
            self.log.emit(f"[OK] Single slice finished in {(time.perf_counter()-t0):.3f}s")
            self.done.emit(esd2d, eac2d, float(t))
        except Exception:
            self.failed.emit(traceback.format_exc())


class FullWorker(QThread):
    log = Signal(str)
    done = Signal()
    failed = Signal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self):
        try:
            self.log.emit("[RUN] Full-angle batch ...")
            run_qt_bsc(self.params)
            self.done.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())


# ---------------- main UI ----------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1250, 780)

        self.cache = {}          # iq2d_obj, bsc_tab, spectrum, etc.
        self.loader = None
        self.worker_single = None
        self.worker_full = None

        self._build()
        self._apply_default_paths()

    # --------- build widgets ---------
    def _build(self):
        root = QVBoxLayout(self)

        # UIUC + URL banner (no SLD wording)
        banner = QLabel(
              "University of Illinois Urbana-Champaign\n"
            "Ultrasound Research Lab – BSC [ESD/EAC] Tool\n"
        )
        banner.setAlignment(Qt.AlignCenter)
        banner.setStyleSheet(
             "background-color: #13294B;"
            "color: white;"
            "font-weight: bold;"
            "padding: 10px;"
            "font-size: 16px;"
            "border-bottom: 3px solid #E84A27;"
        )
        root.addWidget(banner)

        # top bar
        top = QHBoxLayout()
        self.device = QComboBox(); self.device.addItems(["cuda", "cpu"])
        self.load_btn = QPushButton("Load Data (object preview + spectrum)")
        self.run_single_btn = QPushButton("Start (Single slice)")
        self.run_full_btn = QPushButton("Start (Full angles)")

        self.preview_dr = QDoubleSpinBox()
        self.preview_dr.setRange(10.0, 120.0)
        self.preview_dr.setValue(60.0)
        self.preview_dr.setSingleStep(5.0)

        top.addWidget(QLabel("Device:")); top.addWidget(self.device)
        top.addSpacing(14)
        top.addWidget(self.load_btn)
        top.addWidget(self.run_single_btn)
        top.addWidget(self.run_full_btn)
        top.addStretch(1)
        top.addWidget(QLabel("DR (dB):")); top.addWidget(self.preview_dr)
        root.addLayout(top)

        # scrollable params section
        params_box = QGroupBox("BSC mode")
        v = QVBoxLayout(params_box)

        self.ref_dir = self._path_row(v, "Reference folder (refl):", is_file=False)
        self.obj_dir = self._path_row(v, "Object folder (refl):", is_file=False)
        self.out_esd = self._path_row(v, "Output ESD folder:", is_file=False)
        self.out_eac = self._path_row(v, "Output EAC folder:", is_file=False)
        self.bsc_txt = self._path_row(v, "Reference phantom BSC table (.txt/.csv, MHz,value):", is_file=True)

        g = QGridLayout()

        self.wl_mm = self._dspin("Wavelength (mm)", 0.05, 2.0, 0.40, 0.001, 3)
        self.block_wl = self._spin("Block (λ)", 5, 200, 25)
        self.overlap = self._dspin("Overlap", 0.0, 0.98, 0.875, 0.001, 3)

        self.fs_mhz = self._dspin("fs (MHz)", 0.1, 200.0, 5.0, 0.1, 3)
        self.nfft = self._spin("NFFT", 64, 4096, 565)
        self.batch = self._spin("batch_slices", 1, 256, 8)

        self.center_mhz = self._dspin("Center freq (MHz) for baseband↔passband", 0.0, 20.0, 3.6, 0.01, 3)

        self.txt_lo = self._spin("TXT band low idx (inclusive)", 0, 1000000, 0)
        self.txt_hi = self._spin("TXT band high idx (EXCLUSIVE)", 1, 1000000, 115)

        self.form = QComboBox(); self.form.addItems(["gaussian", "faran"])
        self.material = QComboBox(); self.material.addItems(["glass", "poly", "fat", "tung", "res", "agar"])

        self.angle_lv = QSpinBox(); self.angle_lv.setRange(1, 60); self.angle_lv.setValue(1)
        self.z_idx = QSpinBox(); self.z_idx.setRange(0, 200000); self.z_idx.setValue(0)

        grid_widgets = [
            self.wl_mm["box"], self.block_wl["box"], self.overlap["box"],
            self.fs_mhz["box"], self.nfft["box"], self.batch["box"],
            self.center_mhz["box"], self.txt_lo["box"], self.txt_hi["box"],
        ]
        for i, w in enumerate(grid_widgets):
            g.addWidget(w, i // 6, i % 6)

        v.addLayout(g)

        fm_box = QGroupBox("Form factor & material")
        fm = QHBoxLayout(fm_box)
        fm.addWidget(QLabel("Form factor:")); fm.addWidget(self.form)
        fm.addSpacing(18)
        fm.addWidget(QLabel("Material:")); fm.addWidget(self.material)
        fm.addStretch(1)
        v.addWidget(fm_box)

        az_box = QGroupBox("Preview / run selection")
        az = QHBoxLayout(az_box)
        az.addWidget(QLabel("Angle (unit1_#):")); az.addWidget(self.angle_lv)
        az.addSpacing(14)
        az.addWidget(QLabel("z index:")); az.addWidget(self.z_idx)
        az.addStretch(1)
        v.addWidget(az_box)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(params_box)
        root.addWidget(scroll, stretch=0)

        # visuals row (always visible)
        self.preview = QLabel("Object preview (load data)")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(QSize(360, 260))
        self.preview.setStyleSheet("border:1px solid #bbb;background:#fafafa;")
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.spectrum = QLabel("Spectrum (baseband) • averaged across columns")
        self.spectrum.setAlignment(Qt.AlignCenter)
        self.spectrum.setMinimumSize(QSize(560, 260))
        self.spectrum.setStyleSheet("border:1px solid #bbb;background:#fafafa;")
        self.spectrum.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        vr = QHBoxLayout()
        vr.addWidget(self.preview, 1)
        vr.addWidget(self.spectrum, 1)
        root.addLayout(vr)

        # log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        root.addWidget(self.log, stretch=1)

        # signals
        self.load_btn.clicked.connect(self._load_data)
        self.run_single_btn.clicked.connect(self._run_single)
        self.run_full_btn.clicked.connect(self._run_full)

        self.preview_dr.valueChanged.connect(self._refresh_preview)

        # idx/fc -> update overlay
        self.txt_lo["spin"].valueChanged.connect(self._redraw_spectrum)
        self.txt_hi["spin"].valueChanged.connect(self._redraw_spectrum)
        self.center_mhz["spin"].valueChanged.connect(self._redraw_spectrum)

        # fs/nfft changed -> recompute spectrum
        self.fs_mhz["spin"].valueChanged.connect(self._recompute_spectrum_if_loaded)
        self.nfft["spin"].valueChanged.connect(self._recompute_spectrum_if_loaded)

    def _apply_default_paths(self):
        self.ref_dir["edit"].setText(DEFAULT_PATHS["ref_dir"])
        self.obj_dir["edit"].setText(DEFAULT_PATHS["obj_dir"])
        self.out_esd["edit"].setText(DEFAULT_PATHS["out_dir_esd"])
        self.out_eac["edit"].setText(DEFAULT_PATHS["out_dir_eac"])
        self.bsc_txt["edit"].setText(DEFAULT_PATHS["BSC_ref_file"])

    # -------- small UI builders --------
    def _path_row(self, parent_vbox: QVBoxLayout, label: str, is_file: bool):
        le = QLineEdit()
        btn = QPushButton("Browse…")
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        row.addWidget(le, 1)
        row.addWidget(btn)

        def browse():
            if is_file:
                p, _ = QFileDialog.getOpenFileName(self, "Select file")
            else:
                p = QFileDialog.getExistingDirectory(self, "Select folder")
            if p:
                le.setText(p)

        btn.clicked.connect(browse)
        parent_vbox.addLayout(row)
        return {"edit": le}

    def _spin(self, title: str, lo: int, hi: int, default: int):
        box = QGroupBox(title)
        sp = QSpinBox()
        sp.setRange(lo, hi)
        sp.setValue(default)
        lay = QHBoxLayout(box)
        lay.addWidget(sp)
        return {"box": box, "spin": sp}

    def _dspin(self, title: str, lo: float, hi: float, default: float, step: float, decimals: int):
        box = QGroupBox(title)
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setValue(float(default))
        sp.setSingleStep(step)
        sp.setDecimals(decimals)
        lay = QHBoxLayout(box)
        lay.addWidget(sp)
        return {"box": box, "spin": sp}

    # -------- logging --------
    def _append(self, s: str):
        ts = time.strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {s}")
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    # -------- spectrum drawing --------
    def _draw_spectrum_pixmap(self, freqs_mhz, mag_db, *,
                              width, height, dr_db=80.0,
                              vlines_baseband=None,
                              hlines_db=None):
        img = QImage(width, height, QImage.Format_RGB32)
        img.fill(QColor(250, 250, 250))
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing, True)

        L, R, T, B = 70, 20, 20, 40
        Wp = width - L - R
        Hp = height - T - B

        p.setPen(QPen(QColor(200, 200, 200), 1))
        p.drawRect(L, T, Wp, Hp)

        y = np.clip(mag_db, -float(dr_db), 0.0)
        x0, x1 = float(freqs_mhz.min()), float(freqs_mhz.max())
        if x0 == x1:
            x0, x1 = -1.0, 1.0

        def X_of(f):
            return L + int(round((f - x0) / (x1 - x0) * Wp))

        def Y_of(db):
            return T + int(round((0.0 - db) / dr_db * Hp))

        span = x1 - x0
        step = 10 ** np.floor(np.log10(span / 6.0))
        for mult in [1, 2, 2.5, 5, 10]:
            if span / (step * mult) <= 6.0:
                step *= mult
                break
        xticks = np.arange(np.ceil(x0 / step) * step, x1 + step * 0.5, step)
        yticks = [-dr_db, -dr_db / 2.0, 0.0]

        p.setPen(QPen(QColor(230, 230, 230), 1))
        for xv in xticks:
            X = X_of(float(xv))
            p.drawLine(X, T, X, T + Hp)
        for yv in yticks:
            Y = Y_of(float(yv))
            p.drawLine(L, Y, L + Wp, Y)

        p.setPen(QPen(QColor(0, 0, 0), 1))
        font = QFont(); font.setPointSize(9); p.setFont(font)
        for xv in xticks:
            X = X_of(float(xv))
            p.drawText(X - 18, T + Hp + 16, f"{xv:.2g}")
        p.drawText(L + Wp - 130, T + Hp + 32, "MHz (baseband)")
        for yv in yticks:
            Y = Y_of(float(yv))
            p.drawText(6, Y + 4, f"{yv:.0f} dB")

        N = len(freqs_mhz)
        stride = max(1, N // max(Wp, 1))
        Xs = L + ((freqs_mhz[::stride] - x0) / (x1 - x0) * Wp)
        Ys = T + ((0 - y[::stride]) / dr_db * Hp)
        poly = QPolygonF([QPointF(float(x), float(y_)) for x, y_ in zip(Xs, Ys)])
        p.setPen(QPen(QColor(30, 60, 160), 2))
        p.drawPolyline(poly)

        # horizontal reference lines + labels
        if hlines_db:
            p.setPen(QPen(QColor(120, 120, 120), 1, Qt.DashLine))
            for lvl in hlines_db:
                if not (-dr_db <= lvl <= 0.0):
                    continue
                Y = Y_of(float(lvl))
                p.drawLine(L, Y, L + Wp, Y)
                p.setPen(QPen(QColor(80, 80, 80), 1))
                p.drawText(L + 6, Y - 2, f"{abs(int(lvl))} dB")
                p.setPen(QPen(QColor(120, 120, 120), 1, Qt.DashLine))

        # vertical band markers (baseband)
        if vlines_baseband and len(vlines_baseband) == 2:
            fL, fH = float(vlines_baseband[0]), float(vlines_baseband[1])
            p.setPen(QPen(QColor(180, 50, 50), 1, Qt.DashLine))
            for fv in [fL, fH]:
                if fv < x0 or fv > x1:
                    continue
                X = X_of(fv)
                p.drawLine(X, T, X, T + Hp)

        p.end()
        return QPixmap.fromImage(img)

    # -------- param packing --------
    def _collect_params(self) -> dict:
        p = dict(
            ref_dir=self.ref_dir["edit"].text(),
            obj_dir=self.obj_dir["edit"].text(),
            out_dir_esd=self.out_esd["edit"].text(),
            out_dir_eac=self.out_eac["edit"].text(),
            BSC_ref_file=self.bsc_txt["edit"].text(),

            wl_m=float(self.wl_mm["spin"].value()) * 1e-3,
            blocksize_wl=int(self.block_wl["spin"].value()),
            overlap_pc=float(self.overlap["spin"].value()),

            fs=float(self.fs_mhz["spin"].value()) * 1e6,   # Hz
            NFFT=int(self.nfft["spin"].value()),
            batch_slices=int(self.batch["spin"].value()),

            # keep stable defaults (not shown in UI)
            att_ref_dB=0.75,
            att_sam_dB=0.75,
            c=1540.0,

            scat_diams_um=np.arange(5, 101, 2, dtype=np.float32),
            form_factor=str(self.form.currentText()),
            material=str(self.material.currentText()),
            device=str(self.device.currentText()),

            # TXT indices (inclusive, exclusive)
            freq_band_low_idx=int(self.txt_lo["spin"].value()),
            freq_band_high_idx=int(self.txt_hi["spin"].value()),

            # backend expects this for its start-index rule; keep a harmless default
            freq_L_MHz=float(3.5045 - 3.6),
        )
        return p

    # -------- actions --------
    def _load_data(self):
        cfg = self._collect_params()

        if not os.path.isdir(cfg["obj_dir"]):
            QMessageBox.critical(self, "Invalid", "Object folder invalid.")
            return
        if not os.path.isfile(cfg["BSC_ref_file"]):
            QMessageBox.critical(self, "Invalid", "BSC TXT file invalid.")
            return

        self.load_btn.setEnabled(False)
        self._append("[UI] Loading object slice + BSC TXT ...")

        cfg2 = dict(
            obj_dir=cfg["obj_dir"],
            BSC_ref_file=cfg["BSC_ref_file"],
            angle_lv=int(self.angle_lv.value()),
            z_idx=int(self.z_idx.value()),
        )

        self.loader = LoadWorker(cfg2)
        self.loader.log.connect(self._append)
        self.loader.done.connect(self._on_loaded)
        self.loader.failed.connect(self._on_load_failed)
        self.loader.start()

    def _on_loaded(self, payload: dict):
        self.cache["iq2d_obj"] = payload["iq2d_obj"]
        self.cache["Nz"] = payload["Nz"]
        self.cache["bsc_tab"] = payload["bsc_tab"]

        self.z_idx.setMaximum(max(0, int(payload["Nz"]) - 1))

        self._refresh_preview()

        fs_hz = float(self.fs_mhz["spin"].value()) * 1e6
        nfft = int(self.nfft["spin"].value())
        freqs_mhz, mag_db = spectrum_from_slice_avg(self.cache["iq2d_obj"], fs_hz, nfft)
        self.cache["spec_freqs_mhz"] = freqs_mhz
        self.cache["spec_mag_db"] = mag_db

        self._append(f"[READY] Loaded object slice. Nz={payload['Nz']}. Loaded BSC TXT rows={payload['bsc_tab'].shape[0]}.")
        self._redraw_spectrum()
        self.load_btn.setEnabled(True)

    def _on_load_failed(self, tb: str):
        self._append("[ERR] Load failed:\n" + tb)
        self.load_btn.setEnabled(True)

    def _refresh_preview(self):
        if "iq2d_obj" not in self.cache:
            return
        try:
            gray = mag_to_db_u8(self.cache["iq2d_obj"], dr_db=float(self.preview_dr.value()))
            h, w = gray.shape
            qimg = QImage(gray.data, w, h, w, QImage.Format_Grayscale8)
            pix = QPixmap.fromImage(qimg.copy()).scaled(
                self.preview.width(), self.preview.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.preview.setPixmap(pix)
        except Exception as e:
            self._append(f"[WARN] Preview failed: {e}")
            self.preview.setText("Preview failed.")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_preview()
        self._redraw_spectrum()

    def _recompute_spectrum_if_loaded(self):
        if "iq2d_obj" not in self.cache:
            return
        fs_hz = float(self.fs_mhz["spin"].value()) * 1e6
        nfft = int(self.nfft["spin"].value())
        freqs_mhz, mag_db = spectrum_from_slice_avg(self.cache["iq2d_obj"], fs_hz, nfft)
        self.cache["spec_freqs_mhz"] = freqs_mhz
        self.cache["spec_mag_db"] = mag_db
        self._redraw_spectrum()

    def _redraw_spectrum(self):
        if "spec_freqs_mhz" not in self.cache or "bsc_tab" not in self.cache:
            return

        freqs = self.cache["spec_freqs_mhz"]
        mag = self.cache["spec_mag_db"]

        tab = self.cache["bsc_tab"]
        f_txt = tab[:, 0].astype(np.float64)

        lo = int(self.txt_lo["spin"].value())
        hi = int(self.txt_hi["spin"].value())  # EXCLUSIVE
        lo = max(0, min(lo, len(f_txt) - 1))
        hi = max(lo + 1, min(hi, len(f_txt)))

        f_pass_lo = float(f_txt[lo])
        f_pass_hi = float(f_txt[hi - 1])  # last included
        fc = float(self.center_mhz["spin"].value())

        # passband -> baseband overlay
        f_base_lo = f_pass_lo - fc
        f_base_hi = f_pass_hi - fc

        pix = self._draw_spectrum_pixmap(
            freqs, mag,
            width=max(self.spectrum.width(), 820),
            height=max(self.spectrum.height(), 260),
            dr_db=max(float(self.preview_dr.value()), 60.0),
            vlines_baseband=(f_base_lo, f_base_hi),
            hlines_db=[-6, -15, -20],
        )
        self.spectrum.setPixmap(pix)

    def _run_single(self):
        params = self._collect_params()

        for k in ["ref_dir", "obj_dir"]:
            if not os.path.isdir(params[k]):
                QMessageBox.critical(self, "Invalid", f"{k} invalid.")
                return
        if not os.path.isfile(params["BSC_ref_file"]):
            QMessageBox.critical(self, "Invalid", "BSC TXT file invalid.")
            return

        os.makedirs(params["out_dir_esd"], exist_ok=True)
        os.makedirs(params["out_dir_eac"], exist_ok=True)

        angle = int(self.angle_lv.value())
        z = int(self.z_idx.value())

        self._append(
            f"[RUN] Single slice: angle={angle}, z={z}, "
            f"TXT band=[{params['freq_band_low_idx']},{params['freq_band_high_idx']})"
        )

        self.worker_single = SingleWorker(params, angle, z, write=False)
        self.worker_single.log.connect(self._append)
        self.worker_single.done.connect(self._on_single_done)
        self.worker_single.failed.connect(self._on_failed)
        self.worker_single.start()

    def _on_single_done(self, esd2d, eac2d, t):
        self._append(f"[DONE] Single slice compute-only {t*1000:.1f} ms (showing ESD preview)")
        try:
            gray = mag_to_db_u8(esd2d.astype(np.float32), dr_db=float(self.preview_dr.value()))
            h, w = gray.shape
            qimg = QImage(gray.data, w, h, w, QImage.Format_Grayscale8)
            pix = QPixmap.fromImage(qimg.copy()).scaled(
                self.preview.width(), self.preview.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.preview.setPixmap(pix)
        except Exception:
            pass

    def _run_full(self):
        params = self._collect_params()
        for k in ["ref_dir", "obj_dir"]:
            if not os.path.isdir(params[k]):
                QMessageBox.critical(self, "Invalid", f"{k} invalid.")
                return
        if not os.path.isfile(params["BSC_ref_file"]):
            QMessageBox.critical(self, "Invalid", "BSC TXT file invalid.")
            return
        os.makedirs(params["out_dir_esd"], exist_ok=True)
        os.makedirs(params["out_dir_eac"], exist_ok=True)

        self._append(
            f"[RUN] Full angles. TXT band=[{params['freq_band_low_idx']},{params['freq_band_high_idx']})"
        )

        self.worker_full = FullWorker(params)
        self.worker_full.log.connect(self._append)
        self.worker_full.done.connect(lambda: self._append("[DONE] Full-angle batch finished."))
        self.worker_full.failed.connect(self._on_failed)
        self.worker_full.start()

    def _on_failed(self, tb: str):
        self._append("[ERR]\n" + tb)
        QMessageBox.critical(self, "Error", "See log for traceback.")


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
