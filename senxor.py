import sys, os, time, json
from dataclasses import dataclass
from typing import Optional, List
from sysmon_helper import attach_sys_monitor
import numpy as np
import cv2 as cv
from PyQt5 import QtCore, QtGui, QtWidgets
from serial.tools import list_ports

try:
    from senxor.mi48 import MI48
    from senxor.utils import data_to_frame, connect_senxor
except Exception:

    def data_to_frame(data, shape, hflip=False):
        arr = (
            np.frombuffer(data, dtype=np.float32)
            if isinstance(data, (bytes, bytearray))
            else np.zeros(shape, np.float32)
        )
        return arr.reshape(shape)

    def connect_senxor(src=None, **kwargs):
        raise RuntimeError("pysenxor not installed")


PALETTES = {
    "HEATED_IRON": cv.COLORMAP_INFERNO,
    "INFERNO": cv.COLORMAP_INFERNO,
    "MAGMA": (
        cv.COLORMAP_MAGMA if hasattr(cv, "COLORMAP_MAGMA") else cv.COLORMAP_INFERNO
    ),
    "TURBO": cv.COLORMAP_TURBO if hasattr(cv, "COLORMAP_TURBO") else cv.COLORMAP_JET,
    "JET": cv.COLORMAP_JET,
    "BONE": cv.COLORMAP_BONE,
    "HOT": cv.COLORMAP_HOT,
    "VIRIDIS": (
        cv.COLORMAP_VIRIDIS if hasattr(cv, "COLORMAP_VIRIDIS") else cv.COLORMAP_JET
    ),
}

SRC_W, SRC_H = 80, 62  # native sensor


def to_u8_percentiles(arr: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    lo, hi = np.percentile(arr, (p_low, p_high))
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            hi = lo + 1.0
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo) * 255.0
    return arr.astype(np.uint8)


def to_u8_manual(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo) * 255.0
    return arr.astype(np.uint8)


def unsharp_u8(
    img_u8: np.ndarray, sigma: float = 0.7, amount: float = 0.8
) -> np.ndarray:
    blur = cv.GaussianBlur(img_u8, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv.addWeighted(img_u8, 1.0 + amount, blur, -amount, 0)


@dataclass
class FilterParams:
    use_bilateral: bool = True
    gauss_sigma: float = 0.6
    bilateral_d: int = 5
    bilateral_sigc: float = 12.0
    bilateral_sigs: float = 12.0
    unsharp_sigma: float = 0.7
    unsharp_amount: float = 0.8


class Viewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.hover_ix: Optional[int] = None
        self.hover_iy: Optional[int] = None
        self.last_frame_float: Optional[np.ndarray] = None
        self.last_frame_u8: Optional[np.ndarray] = None
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        self.sensor_fps = 0.0
        self._prev_fc = None
        self._prev_ts = None
        self._prev_wall = None

        # Sensor FPS (robust, wall-time window)
        self._sens_win_frames = 0
        self._sens_win_t0 = time.time()
        self._sensor_fps_ema = None
        self._loading_settings = False

        self.setWindowTitle("SenXor Viewer")
        self.setMinimumSize(980, 720)

        self.mi48: Optional[MI48] = None
        self.connected_port: Optional[str] = None
        self.port_candidates: List[str] = []
        self.target_fps = 25

        self.temporal: Optional[np.ndarray] = None
        self.alpha_temporal = 0.20

        self.src_w = SRC_W
        self.src_h = SRC_H

        self.p_low = 5.0
        self.p_high = 99.5
        self.min_manual = 31.7
        self.max_manual = 36.7

        self.filt = FilterParams()

        self.clahe_clip = 2.0
        self.clahe_tiles_x = 8
        self.clahe_tiles_y = 6

        self.palette_name = "HEATED_IRON"

        # Auto behaviors
        self.auto_connect = True
        self.auto_start = True

        self.writer: Optional[cv.VideoWriter] = None
        self.raw_win = None  # Raw sensor live window
        self.recording = False
        self.record_fps = 15

        self.scale_factor = 6
        self.render_size = (SRC_W * self.scale_factor, SRC_H * self.scale_factor)

        base_dir = os.path.dirname(
            getattr(sys.modules[__name__], "__file__", sys.argv[0]) or os.getcwd()
        )
        self.output_dir = os.path.join(base_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)

        self.rotate_mode = "0°"

        # Fullscreen toggle state
        self.fullscreen_mode = False
        self._prev_geom = None
        self._prev_is_max = False
        self.flip_h = False
        self.flip_v = False

        self._save_debounce = QtCore.QTimer(self)
        self._save_debounce.setSingleShot(True)
        self._save_debounce.setInterval(150)
        self._save_debounce.timeout.connect(self.save_settings)

        self.settings_path = os.path.join(base_dir, "settings.json")

        self._build_ui()

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(0)
        self.timer.timeout.connect(self.read_frame)

        QtCore.QTimer.singleShot(0, self.post_init)

    def post_init(self):
        try:
            self.refresh_ports()
        except Exception as e:
            print("[post_init] refresh_ports error:", e)
        try:
            self._loading_settings = True
            ok = self.load_settings()
            print("[post_init] load_settings ok=", ok)
        except Exception as e:
            print("[post_init] load_settings error:", e)
        finally:
            self._loading_settings = False
        # Auto behaviors after settings are loaded
        try:
            if self.auto_connect:
                # If saved port is present, it's already selected by load_settings
                port_txt = self.port_combo.currentText()
                if port_txt and port_txt != "(no ports)":
                    self.do_connect(silent=True)
            if self.auto_start and self.mi48 is not None:
                self.start_stream(silent=True)
        except Exception as e:
            try:
                self.statusBar().showMessage(f"Auto start failed: {e}", 5000)
            except Exception:
                pass

    def _hbox(self, widgets):
        w = QtWidgets.QWidget()
        l = QtWidgets.QHBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(6)
        l.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        for x in widgets:
            l.addWidget(x)
        # Keep the whole row's field area hugging the left
        w.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        return w

    def _btn(self, text, slot):
        b = QtWidgets.QPushButton(text)
        b.clicked.connect(slot)
        return b

    def mark_dirty(self, *args, **kwargs):
        if self._loading_settings:
            return
        try:
            self._save_debounce.start()
        except Exception:
            try:
                self.save_settings()
            except Exception:
                pass

    def _build_ui(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # LEFT
        left = QtWidgets.QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(0)
        root.addLayout(left, 0)

        # IMAGE
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setStyleSheet("background:black")
        self.image_label.setMouseTracking(True)
        self.image_label.setFixedSize(*self.render_size)
        self.image_label.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        left.addWidget(self.image_label, stretch=0)
        self.image_label.installEventFilter(self)

        # INFO BAR
        self.info_target = QtWidgets.QLabel("Temperature at (x,y): -- °C")
        self.info_die = QtWidgets.QLabel("VDD: -- V  |  Target: -- °C  |  Die: -- °C")
        self.info_header = QtWidgets.QLabel(
            "frame: -- | ts: -- | min: -- | max: -- | crc: --"
        )
        small_css = "font-family: Consolas, 'Courier New', monospace; font-size: 11px; margin:0px; padding:0px;"
        for _lbl in (self.info_target, self.info_die, self.info_header):
            _lbl.setStyleSheet(small_css)
            _lbl.setWordWrap(False)
            _lbl.setContentsMargins(0, 0, 0, 0)
            _lbl.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        info_box = QtWidgets.QWidget()
        info_layout = QtWidgets.QVBoxLayout(info_box)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(0)
        info_layout.addWidget(self.info_target)
        info_layout.addWidget(self.info_die)
        info_layout.addWidget(self.info_header)
        info_box.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        left.addWidget(info_box, stretch=0)

        # RIGHT (canonical block)
        right_scroll = QtWidgets.QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(320)
        root.addWidget(right_scroll, 1)
        self.right_scroll = right_scroll

        right = QtWidgets.QWidget()
        right_scroll.setWidget(right)
        form = QtWidgets.QFormLayout(right)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(4)
        form.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        form.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        # 30% (left) / 70% (right)
        root.setStretch(0, 3)
        root.setStretch(1, 7)

        # 30% (left) / 70% (right)

        # Serial row
        self.port_combo = QtWidgets.QComboBox()
        self.port_combo.currentTextChanged.connect(self.mark_dirty)
        form.addRow(
            "Serial Port",
            self._hbox(
                [
                    self.port_combo,
                    self._btn("Refresh", self.refresh_ports),
                    self._btn("Connect", self.do_connect),
                    self._btn("Disconnect", self.do_disconnect),
                ]
            ),
        )

        # Stream row
        self.btn_start = self._btn("Start", self.start_stream)
        self.btn_stop = self._btn("Stop", self.stop_stream)
        self.btn_stop.setEnabled(False)
        self.spin_target_fps = QtWidgets.QSpinBox()
        self.spin_target_fps.setRange(1, 25)
        self.spin_target_fps.setValue(self.target_fps)
        self.spin_target_fps.valueChanged.connect(self.on_target_fps_changed)
        self.spin_target_fps.valueChanged.connect(self.mark_dirty)
        self.btn_fps_dn = self._btn("Down", self.fps_down)
        self.btn_fps_up = self._btn("Up", self.fps_up)
        self.lbl_fps = QtWidgets.QLabel(
            f"FPS: 0.00 (target: {self.target_fps}, sensor: 0.00)"
        )
        form.addRow(
            "Stream",
            self._hbox(
                [
                    self.btn_start,
                    self.btn_stop,
                    QtWidgets.QLabel("Target"),
                    self.spin_target_fps,
                    self.btn_fps_dn,
                    self.btn_fps_up,
                    self.lbl_fps,
                ]
            ),
        )

        # Auto options
        self.chk_auto_connect = QtWidgets.QCheckBox("Auto-Connect")
        self.chk_auto_connect.setChecked(True)
        self.chk_auto_connect.toggled.connect(self.on_auto_flags_changed)
        self.chk_auto_start = QtWidgets.QCheckBox("Auto-Start")
        self.chk_auto_start.setChecked(True)
        self.chk_auto_start.toggled.connect(self.on_auto_flags_changed)
        form.addRow(
            "Automation", self._hbox([self.chk_auto_connect, self.chk_auto_start])
        )

        # Palette
        self.palette_combo = QtWidgets.QComboBox()
        self.palette_combo.addItems(list(PALETTES.keys()))
        self.palette_combo.setCurrentText(self.palette_name)
        self.palette_combo.currentTextChanged.connect(self.on_palette)
        self.palette_combo.currentTextChanged.connect(self.mark_dirty)
        form.addRow("Palette", self._hbox([self.palette_combo]))

        # Rotate / Flip
        self.rotate_combo = QtWidgets.QComboBox()
        self.rotate_combo.addItems(["0°", "90° CW", "180°", "270° CW"])
        self.rotate_combo.currentTextChanged.connect(self.on_rotate_change)
        self.rotate_combo.currentTextChanged.connect(self.mark_dirty)
        self.chk_flip_h = QtWidgets.QCheckBox("Flip H")
        self.chk_flip_h.toggled.connect(self.on_flip_change)
        self.chk_flip_h.toggled.connect(self.mark_dirty)
        self.chk_flip_v = QtWidgets.QCheckBox("Flip V")
        self.chk_flip_v.toggled.connect(self.on_flip_change)
        self.chk_flip_v.toggled.connect(self.mark_dirty)
        form.addRow(
            "Rotate / Flip",
            self._hbox([self.rotate_combo, self.chk_flip_h, self.chk_flip_v]),
        )

        # Min/Max markers + values
        self.chk_minmax = QtWidgets.QCheckBox("Show Min/Max markers")
        self.chk_minmax.setChecked(True)
        self.chk_minmax.toggled.connect(self.mark_dirty)
        self.chk_minmax_vals = QtWidgets.QCheckBox("Show values (°C)")
        self.chk_minmax_vals.setChecked(True)
        self.chk_minmax_vals.toggled.connect(self.mark_dirty)
        form.addRow("Markers", self._hbox([self.chk_minmax, self.chk_minmax_vals]))

        # Scaling
        self.chk_auto = QtWidgets.QCheckBox("Auto (percentiles)")
        self.chk_auto.setChecked(True)
        self.chk_auto.toggled.connect(self.on_scale_mode)
        self.chk_auto.toggled.connect(self.mark_dirty)
        form.addRow("Scaling", self._hbox([self.chk_auto]))

        self.spin_p_low = QtWidgets.QDoubleSpinBox()
        self.spin_p_low.setRange(0.0, 50.0)
        self.spin_p_low.setDecimals(1)
        self.spin_p_low.setValue(self.p_low)
        self.spin_p_low.valueChanged.connect(self.mark_dirty)
        self.spin_p_high = QtWidgets.QDoubleSpinBox()
        self.spin_p_high.setRange(50.0, 100.0)
        self.spin_p_high.setDecimals(1)
        self.spin_p_high.setValue(self.p_high)
        self.spin_p_high.valueChanged.connect(self.mark_dirty)
        form.addRow(
            "Percentiles (lo/hi)", self._hbox([self.spin_p_low, self.spin_p_high])
        )

        self.spin_min = QtWidgets.QDoubleSpinBox()
        self.spin_min.setRange(-50.0, 150.0)
        self.spin_min.setDecimals(2)
        self.spin_min.setValue(self.min_manual)
        self.spin_min.valueChanged.connect(self.mark_dirty)
        self.spin_max = QtWidgets.QDoubleSpinBox()
        self.spin_max.setRange(-50.0, 150.0)
        self.spin_max.setDecimals(2)
        self.spin_max.setValue(self.max_manual)
        self.spin_max.valueChanged.connect(self.mark_dirty)
        form.addRow("Manual (min/max °C)", self._hbox([self.spin_min, self.spin_max]))

        # CLAHE
        self.chk_clahe = QtWidgets.QCheckBox("Use CLAHE")
        self.chk_clahe.setChecked(True)
        self.chk_clahe.toggled.connect(self.mark_dirty)
        self.spin_clip = QtWidgets.QDoubleSpinBox()
        self.spin_clip.setRange(0.5, 6.0)
        self.spin_clip.setDecimals(1)
        self.spin_clip.setSingleStep(0.1)
        self.spin_clip.setValue(self.clahe_clip)
        self.spin_clip.valueChanged.connect(self.mark_dirty)
        self.spin_tiles_x = QtWidgets.QSpinBox()
        self.spin_tiles_x.setRange(2, 16)
        self.spin_tiles_x.setValue(self.clahe_tiles_x)
        self.spin_tiles_x.valueChanged.connect(self.mark_dirty)
        self.spin_tiles_y = QtWidgets.QSpinBox()
        self.spin_tiles_y.setRange(2, 16)
        self.spin_tiles_y.setValue(self.clahe_tiles_y)
        self.spin_tiles_y.valueChanged.connect(self.mark_dirty)
        form.addRow(
            "CLAHE",
            self._hbox(
                [
                    self.chk_clahe,
                    QtWidgets.QLabel("clip"),
                    self.spin_clip,
                    QtWidgets.QLabel("tiles X/Y"),
                    self.spin_tiles_x,
                    self.spin_tiles_y,
                ]
            ),
        )

        # Filters
        self.chk_bilat = QtWidgets.QCheckBox("Bilateral")
        self.chk_bilat.setChecked(self.filt.use_bilateral)
        self.chk_bilat.toggled.connect(self.mark_dirty)
        self.spin_bilat_d = QtWidgets.QSpinBox()
        self.spin_bilat_d.setRange(1, 15)
        self.spin_bilat_d.setValue(self.filt.bilateral_d)
        self.spin_bilat_d.valueChanged.connect(self.mark_dirty)
        self.spin_bilat_sc = QtWidgets.QDoubleSpinBox()
        self.spin_bilat_sc.setRange(1.0, 50.0)
        self.spin_bilat_sc.setDecimals(1)
        self.spin_bilat_sc.setSingleStep(0.1)
        self.spin_bilat_sc.setValue(self.filt.bilateral_sigc)
        self.spin_bilat_sc.valueChanged.connect(self.mark_dirty)
        self.spin_bilat_ss = QtWidgets.QDoubleSpinBox()
        self.spin_bilat_ss.setRange(1.0, 50.0)
        self.spin_bilat_ss.setDecimals(1)
        self.spin_bilat_ss.setSingleStep(0.1)
        self.spin_bilat_ss.setValue(self.filt.bilateral_sigs)
        self.spin_bilat_ss.valueChanged.connect(self.mark_dirty)
        form.addRow(
            "Bilateral (d/σC/σS)",
            self._hbox(
                [
                    self.chk_bilat,
                    self.spin_bilat_d,
                    self.spin_bilat_sc,
                    self.spin_bilat_ss,
                ]
            ),
        )

        self.spin_gauss = QtWidgets.QDoubleSpinBox()
        self.spin_gauss.setRange(0.0, 3.0)
        self.spin_gauss.setDecimals(2)
        self.spin_gauss.setSingleStep(0.1)
        self.spin_gauss.setValue(self.filt.gauss_sigma)
        self.spin_gauss.valueChanged.connect(self.mark_dirty)
        form.addRow("Gaussian σ", self._hbox([self.spin_gauss]))

        self.spin_unsharp_sigma = QtWidgets.QDoubleSpinBox()
        self.spin_unsharp_sigma.setRange(0.0, 3.0)
        self.spin_unsharp_sigma.setDecimals(2)
        self.spin_unsharp_sigma.setSingleStep(0.1)
        self.spin_unsharp_sigma.setValue(self.filt.unsharp_sigma)
        self.spin_unsharp_sigma.valueChanged.connect(self.mark_dirty)
        self.spin_unsharp_amt = QtWidgets.QDoubleSpinBox()
        self.spin_unsharp_amt.setRange(0.0, 2.0)
        self.spin_unsharp_amt.setDecimals(2)
        self.spin_unsharp_amt.setSingleStep(0.1)
        self.spin_unsharp_amt.setValue(self.filt.unsharp_amount)
        self.spin_unsharp_amt.valueChanged.connect(self.mark_dirty)
        form.addRow(
            "Unsharp (σ, amount)",
            self._hbox([self.spin_unsharp_sigma, self.spin_unsharp_amt]),
        )

        self.scale_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scale_slider.setMinimum(4)
        self.scale_slider.setMaximum(12)
        self.scale_slider.setSingleStep(1)
        self.scale_slider.setValue(self.scale_factor)
        self.scale_slider.valueChanged.connect(self.on_scale_factor_changed)
        self.scale_slider.valueChanged.connect(self.mark_dirty)
        form.addRow("Scale (×)", self.scale_slider)

        self.out_dir_edit = QtWidgets.QLineEdit(self.output_dir)
        self.out_dir_edit.setObjectName("output_dir_edit")
        self.out_dir_edit.textChanged.connect(self.mark_dirty)
        self.btn_browse = self._btn("…", self.choose_dir)
        form.addRow("Output dir", self._hbox([self.out_dir_edit, self.btn_browse]))

        self.btn_snapshot = self._btn("Snapshot (PNG)", self.do_snapshot)
        self.btn_record = QtWidgets.QPushButton("Start Recording")
        self.btn_record.clicked.connect(self.toggle_recording)
        self.btn_raw = self._btn("Raw sensor view", self.open_raw_window)
        form.addRow(
            "Capture", self._hbox([self.btn_snapshot, self.btn_record, self.btn_raw])
        )

        self.btn_perf = self._btn("Max FPS preset", self.apply_perf_preset)
        form.addRow("Performance", self._hbox([self.btn_perf]))

        self.lbl_settings_path = QtWidgets.QLabel(self.settings_path)
        self.btn_save_settings = self._btn("Save settings", self._save_settings_click)
        self.btn_load_settings = self._btn("Load settings", self._load_settings_click)
        self.btn_open_settings = self._btn("Open folder", self._open_settings_folder)
        form.addRow(
            "Settings",
            self._hbox(
                [
                    self.lbl_settings_path,
                    self.btn_save_settings,
                    self.btn_load_settings,
                    self.btn_open_settings,
                ]
            ),
        )
        attach_sys_monitor(self)
        # Ensure status bar exists
        try:
            self.statusBar()
        except Exception:
            pass
        # Apply compact tweaks on the right pane
        try:
            self._apply_compact_ui(right)
        except Exception:
            pass

    def eventFilter(self, obj, event):
        if obj is self.image_label and event.type() == QtCore.QEvent.MouseMove:
            self.on_mouse_move(event)
            return True
        if obj is self.image_label and event.type() == QtCore.QEvent.MouseButtonPress:
            try:
                self.toggle_fullscreen_image()
            except Exception:
                pass
            return True
            self.on_mouse_move(event)
            return True
        return super().eventFilter(obj, event)

    def toggle_fullscreen_image(self):
        """Toggle app full screen and hide/show the options panel. The image stays centered."""
        try:
            # Ensure status bar exists (avoid accidental popups on Windows)
            self.statusBar()
        except Exception:
            pass

        if not self.fullscreen_mode:
            # Enter fullscreen: store state, hide options, go full
            try:
                self._prev_geom = self.saveGeometry()
            except Exception:
                self._prev_geom = None
            self._prev_is_max = bool(self.windowState() & QtCore.Qt.WindowMaximized)

            # Hide the right options panel
            try:
                if hasattr(self, "right_scroll") and self.right_scroll is not None:
                    self.right_scroll.hide()
            except Exception:
                pass

            # Go fullscreen
            try:
                self.showFullScreen()
            except Exception:
                self.showMaximized()

            # Keep image centered; we already align center and keep fixed size
            self.fullscreen_mode = True
            try:
                self.statusBar().showMessage(
                    "Fullscreen on (click image to exit).", 2000
                )
            except Exception:
                pass

        else:
            # Leave fullscreen: show options and restore geometry/state
            try:
                self.showNormal()
                if self._prev_is_max:
                    self.showMaximized()
                if self._prev_geom is not None:
                    self.restoreGeometry(self._prev_geom)
            except Exception:
                pass
            try:
                if hasattr(self, "right_scroll") and self.right_scroll is not None:
                    self.right_scroll.show()
            except Exception:
                pass
            self.fullscreen_mode = False
            try:
                self.statusBar().showMessage("Fullscreen off.", 2000)
            except Exception:
                pass

    def on_palette(self, name: str):
        self.palette_name = name

    def on_scale_mode(self, *_):
        pass

    def on_auto_flags_changed(self, *_):
        self.auto_connect = bool(self.chk_auto_connect.isChecked())
        self.auto_start = bool(self.chk_auto_start.isChecked())
        self.mark_dirty()

    def on_scale_factor_changed(self, val: int):
        self.scale_factor = max(1, int(val))
        w, h = self.get_render_size_current()
        self.image_label.setFixedSize(w, h)
        self.image_label.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        self._sync_left_width()
        self._sync_left_width()

    def on_rotate_change(self, mode: str):
        self.rotate_mode = mode
        w, h = self.get_render_size_current()
        self.image_label.setFixedSize(w, h)
        self.image_label.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        self._sync_left_width()

    def on_flip_change(self, *_):
        self.flip_h = self.chk_flip_h.isChecked()
        self.flip_v = self.chk_flip_v.isChecked()

    def apply_perf_preset(self):
        try:
            self.chk_clahe.setChecked(False)
        except Exception:
            pass
        try:
            self.chk_bilat.setChecked(False)
        except Exception:
            pass
        try:
            self.spin_gauss.setValue(0.0)
        except Exception:
            pass
        try:
            self.spin_unsharp_sigma.setValue(0.0)
            self.spin_unsharp_amt.setValue(0.0)
        except Exception:
            pass
        try:
            self.scale_slider.setValue(4)
        except Exception:
            pass
        try:
            self.spin_target_fps.setValue(25)
        except Exception:
            pass
        self.mark_dirty()

    def _save_settings_click(self):
        ok = self.save_settings()
        QtWidgets.QMessageBox.information(
            self, "Settings", "Saved." if ok else "Save failed."
        )

    def _load_settings_click(self):
        ok = self.load_settings()
        QtWidgets.QMessageBox.information(
            self, "Settings", "Loaded." if ok else "Load failed."
        )

    def _open_settings_folder(self):
        folder = os.path.dirname(self.settings_path)
        try:
            if sys.platform.startswith("win"):
                os.startfile(folder)  # type: ignore
            elif sys.platform == "darwin":
                import subprocess

                subprocess.Popen(["open", folder])
            else:
                import subprocess

                subprocess.Popen(["xdg-open", folder])
        except Exception:
            QtWidgets.QMessageBox.warning(self, "Open folder", folder)

    def image_coords_from_label_pos(self, px: int, py: int):
        pm = self.image_label.pixmap()
        if pm is None:
            return None, None
        pm_w = pm.width()
        pm_h = pm.height()
        lb_w = self.image_label.width()
        lb_h = self.image_label.height()
        off_x = max(0, (lb_w - pm_w) // 2)
        off_y = max(0, (lb_h - pm_h) // 2)
        x_in_pm = px - off_x
        y_in_pm = py - off_y
        if x_in_pm < 0 or y_in_pm < 0 or x_in_pm >= pm_w or y_in_pm >= pm_h:
            return None, None
        ix = int(x_in_pm / pm_w * self.src_w)
        iy = int(y_in_pm / pm_h * self.src_h)
        ix = max(0, min(self.src_w - 1, ix))
        iy = max(0, min(self.src_h - 1, iy))
        return ix, iy

    def on_mouse_move(self, event: QtGui.QMouseEvent):
        ix, iy = self.image_coords_from_label_pos(event.pos().x(), event.pos().y())
        if ix is None:
            self.hover_ix = None
            self.hover_iy = None
            self.info_target.setText("Temperature at (x,y): -- °C")
            return
        self.hover_ix, self.hover_iy = ix, iy
        if self.last_frame_float is not None:
            val = float(self.last_frame_float[iy, ix])
            self.info_target.setText(f"Temperature at ({ix},{iy}) is {val:.2f}°C")

    def refresh_ports(self):
        self.port_combo.clear()
        ports = [p.device for p in list_ports.comports()]
        if not ports:
            self.port_combo.addItem("(no ports)")
        for p in ports:
            self.port_combo.addItem(p)

    def do_connect(self, silent: bool = False):
        port = self.port_combo.currentText()
        if not port or port == "(no ports)":
            QtWidgets.QMessageBox.warning(self, "No port", "No serial ports detected.")
            return
        try:
            self.mi48, self.connected_port, self.port_candidates = connect_senxor(
                src=port
            )
            try:
                self.mi48.set_fps(min(25, self.target_fps or 25))
            except Exception:
                pass
            # No popup; write in status bar
            try:
                self.statusBar().showMessage(
                    f"Connected on {self.connected_port}", 3000
                )
            except Exception:
                pass
            self.mark_dirty()
        except Exception as e:
            if not silent:
                try:
                    self.statusBar().showMessage(f"Connect failed: {e}", 5000)
                except Exception:
                    pass

    def do_disconnect(self):
        self.stop_stream()
        try:
            if self.mi48:
                self.mi48.stop()
        except Exception:
            pass
        self.mi48 = None
        self.connected_port = None
        try:
            self.statusBar().showMessage("Device disconnected.", 3000)
        except Exception:
            pass
        self.mark_dirty()

    def start_stream(self, silent: bool = False):
        if not self.mi48:
            QtWidgets.QMessageBox.warning(
                self, "Not connected", "Connect the device first."
            )
            return
        try:
            self.mi48.start(stream=True, with_header=True)
            if self.spin_target_fps.value() != self.target_fps:
                self.target_fps = int(self.spin_target_fps.value())
            try:
                self.mi48.set_fps(min(25, self.target_fps or 25))
            except Exception:
                pass
            self.temporal = None
            self.frame_count = 0
            self.last_fps_time = time.time()
            self.timer.start()
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.mark_dirty()
        except Exception as e:
            if not silent:
                try:
                    self.statusBar().showMessage(f"Start failed: {e}", 5000)
                except Exception:
                    pass

    def stop_stream(self):
        self.timer.stop()
        try:
            if self.mi48:
                self.mi48.stop()
        except Exception:
            pass
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self.recording:
            self.stop_recording()
        self.mark_dirty()

    def on_target_fps_changed(self, val: int):
        self.target_fps = max(1, min(25, int(val)))
        try:
            if self.mi48:
                self.mi48.set_fps(self.target_fps)
        except Exception:
            pass
        try:
            self.lbl_fps.setText(
                f"FPS: {self.current_fps:.2f} (target: {self.target_fps}, sensor: {self.sensor_fps:.2f})"
            )
        except Exception:
            pass
        self.mark_dirty()

    def fps_up(self):
        self.spin_target_fps.setValue(min(25, self.spin_target_fps.value() + 1))

    def fps_down(self):
        self.spin_target_fps.setValue(max(1, self.spin_target_fps.value() - 1))

    def choose_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose output directory", self.out_dir_edit.text()
        )
        if d:
            self.out_dir_edit.setText(d)
            self.output_dir = d
            self.mark_dirty()

    def do_snapshot(self):
        if self.last_frame_u8 is None:
            QtWidgets.QMessageBox.warning(self, "No frame", "No frame to save yet.")
            return
        os.makedirs(self.output_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        cmap = PALETTES.get(self.palette_name, cv.COLORMAP_INFERNO)
        vis = cv.applyColorMap(self.last_frame_u8, cmap)
        w, h = self.get_render_size_current()
        vis = cv.resize(vis, (w, h), interpolation=cv.INTER_LANCZOS4)
        path = os.path.join(self.output_dir, f"senxor_{ts}.png")
        cv.imwrite(path, vis)
        QtWidgets.QMessageBox.information(self, "Saved", "Snapshot saved: " + path)

    def toggle_recording(self):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        os.makedirs(self.output_dir, exist_ok=True)
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"senxor_{ts}.mp4")
        w, h = self.get_render_size_current()
        self.writer = cv.VideoWriter(path, fourcc, self.record_fps, (w, h))
        if not self.writer.isOpened():
            self.writer = None
            QtWidgets.QMessageBox.critical(
                self, "Record", "Failed to open VideoWriter. Check codecs."
            )
            return
        self.recording = True
        self.btn_record.setText("Stop Recording")

    def stop_recording(self):
        if self.writer is not None:
            try:
                self.writer.release()
            except Exception:
                pass
            self.writer = None
        self.recording = False
        self.btn_record.setText("Start Recording")

    def apply_rotation_flip(self, arr: np.ndarray) -> np.ndarray:
        out = arr
        mode = self.rotate_mode
        if mode == "0°":
            pass
        elif mode == "90° CW":
            out = np.rot90(out, k=3)
        elif mode == "180°":
            out = np.rot90(out, k=2)
        elif mode == "270° CW":
            out = np.rot90(out, k=1)
        if self.flip_h:
            out = np.fliplr(out)
        if self.flip_v:
            out = np.flipud(out)
        self.src_h, self.src_w = out.shape[:2]
        return out

    def get_render_size_current(self):
        if self.rotate_mode in ("90° CW", "270° CW"):
            w = SRC_H * self.scale_factor
            h = SRC_W * self.scale_factor
        else:
            w = SRC_W * self.scale_factor
            h = SRC_H * self.scale_factor
        return (w, h)

    def _apply_compact_ui(self, container: QtWidgets.QWidget):
        # Global compact stylesheet for the right pane
        css = (
            "* { font-size: 11px; }"
            " QLabel { margin: 0px; padding: 0px; }"
            " QPushButton { padding: 2px 6px; }"
            " QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox { padding: 1px 3px; }"
        )
        try:
            container.setStyleSheet(container.styleSheet() + css)
        except Exception:
            container.setStyleSheet(css)

        def tune(w):
            if isinstance(w, QtWidgets.QPushButton):
                w.setFixedHeight(24)
                w.setMinimumWidth(0)
                w.setMaximumWidth(130)
                w.setSizePolicy(
                    QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed
                )
            if isinstance(w, QtWidgets.QComboBox):
                w.setMinimumWidth(80)
                w.setMaximumWidth(170)
                w.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
            if isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                w.setMinimumWidth(70)
                w.setMaximumWidth(110)
            if isinstance(w, QtWidgets.QLineEdit):
                name = w.objectName().lower()
                if "output" in name or "dir" in name:
                    w.setMinimumWidth(160)
                    w.setSizePolicy(
                        QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
                    )
                else:
                    w.setMaximumWidth(220)
            if isinstance(w, QtWidgets.QLabel):
                w.setWordWrap(False)

        def walk(widget):
            tune(widget)
            for child in widget.findChildren(QtWidgets.QWidget):
                tune(child)

        walk(container)

    def _sync_left_width(self):
        w, h = self.get_render_size_current()
        # Fix label and info widths to image width to avoid pushing right pane
        self.image_label.setFixedSize(w, h)
        for lbl in (self.info_target, self.info_die, self.info_header):
            lbl.setFixedWidth(w)
            # Elide text to fit available width
            try:
                fm = lbl.fontMetrics()
                txt = lbl.text()
                el = fm.elidedText(txt, QtCore.Qt.ElideRight, max(10, w - 12))
                if el != txt:
                    lbl.setText(el)
            except Exception:
                pass

    def read_frame(self):
        header = None
        data = None
        try:
            data, header = self.mi48.read()
        except Exception:
            return
        if data is None:
            return

        base = data_to_frame(data, (SRC_W, SRC_H), hflip=False).astype(np.float32)
        # Keep raw frame exactly as sensor provides (no rotation, no filters)
        self.last_raw_float = base.copy()
        try:
            lo = float(np.min(self.last_raw_float))
            hi = float(np.max(self.last_raw_float))
            if hi <= lo:
                hi = lo + 1.0
            raw_norm = np.clip(
                (self.last_raw_float - lo) / (hi - lo) * 255.0, 0, 255
            ).astype(np.uint8)
        except Exception:
            raw_norm = np.zeros_like(base, dtype=np.uint8)
        self.last_raw_u8 = raw_norm

        self.alpha_temporal = max(0.0, min(1.0, self.alpha_temporal))
        if self.temporal is None:
            self.temporal = base.copy()
        else:
            self.temporal = (
                1.0 - self.alpha_temporal
            ) * self.temporal + self.alpha_temporal * base

        if self.chk_auto.isChecked():
            img8 = to_u8_percentiles(
                self.temporal,
                float(self.spin_p_low.value()),
                float(self.spin_p_high.value()),
            )
        else:
            img8 = to_u8_manual(
                self.temporal,
                float(self.spin_min.value()),
                float(self.spin_max.value()),
            )

        if self.chk_clahe.isChecked():
            clip = float(self.spin_clip.value())
            tiles = (int(self.spin_tiles_x.value()), int(self.spin_tiles_y.value()))
            clahe = cv.createCLAHE(clipLimit=clip, tileGridSize=tiles)
            img8 = clahe.apply(img8)

        if self.chk_bilat.isChecked():
            img8 = cv.bilateralFilter(
                img8,
                d=int(self.spin_bilat_d.value()),
                sigmaColor=float(self.spin_bilat_sc.value()),
                sigmaSpace=float(self.spin_bilat_ss.value()),
            )
        if float(self.spin_gauss.value()) > 0.0:
            img8 = cv.GaussianBlur(img8, (0, 0), float(self.spin_gauss.value()))
        if float(self.spin_unsharp_amt.value()) > 0.0:
            img8 = unsharp_u8(
                img8,
                float(self.spin_unsharp_sigma.value()),
                float(self.spin_unsharp_amt.value()),
            )

        img8_rot = self.apply_rotation_flip(img8)
        temp_rot = self.apply_rotation_flip(self.temporal.copy())

        self.last_frame_u8 = img8_rot
        self.last_frame_float = temp_rot

        cmap = PALETTES.get(self.palette_name, cv.COLORMAP_INFERNO)
        vis = cv.applyColorMap(img8_rot, cmap)

        w, h = self.get_render_size_current()
        vis = cv.resize(vis, (w, h), interpolation=cv.INTER_NEAREST)

        if self.hover_ix is not None and self.hover_iy is not None:
            sx = max(1, int(round(w / self.src_w)))
            sy = max(1, int(round(h / self.src_h)))
            x0 = int(self.hover_ix) * sx
            y0 = int(self.hover_iy) * sy
            cv.rectangle(
                vis, (x0, y0), (x0 + sx - 1, y0 + sy - 1), (0, 255, 0), thickness=-1
            )

        if self.chk_minmax.isChecked() and self.last_frame_float is not None:
            arr = self.last_frame_float
            try:
                idx_max = int(np.nanargmax(arr))
                idx_min = int(np.nanargmin(arr))
                my, mx = np.unravel_index(idx_max, arr.shape)
                ny, nx = np.unravel_index(idx_min, arr.shape)

                sx = max(1, int(round(w / self.src_w)))
                sy = max(1, int(round(h / self.src_h)))
                hot_x0, hot_y0 = int(mx) * sx, int(my) * sy
                cold_x0, cold_y0 = int(nx) * sx, int(ny) * sy

                cv.rectangle(
                    vis,
                    (hot_x0, hot_y0),
                    (hot_x0 + sx - 1, hot_y0 + sy - 1),
                    (0, 0, 0),
                    thickness=1,
                )
                cv.rectangle(
                    vis,
                    (cold_x0, cold_y0),
                    (cold_x0 + sx - 1, cold_y0 + sy - 1),
                    (0, 0, 0),
                    thickness=1,
                )
                if sx > 2 and sy > 2:
                    cv.rectangle(
                        vis,
                        (hot_x0 + 1, hot_y0 + 1),
                        (hot_x0 + sx - 2, hot_y0 + sy - 2),
                        (0, 0, 255),
                        thickness=-1,
                    )
                    cv.rectangle(
                        vis,
                        (cold_x0 + 1, cold_y0 + 1),
                        (cold_x0 + sx - 2, cold_y0 + sy - 2),
                        (255, 255, 0),
                        thickness=-1,
                    )
                else:
                    cv.rectangle(
                        vis,
                        (hot_x0, hot_y0),
                        (hot_x0 + sx - 1, hot_y0 + sy - 1),
                        (0, 0, 255),
                        thickness=-1,
                    )
                    cv.rectangle(
                        vis,
                        (cold_x0, cold_y0),
                        (cold_x0 + sx - 1, cold_y0 + sy - 1),
                        (255, 255, 0),
                        thickness=-1,
                    )

                if self.chk_minmax_vals.isChecked():
                    hot_val = float(arr[my, mx])
                    cold_val = float(arr[ny, nx])
                    hot_txt = f"{hot_val:.2f} C"
                    cold_txt = f"{cold_val:.2f} C"
                    tx_hot = min(hot_x0 + sx + 4, w - 1)
                    ty_hot = min(hot_y0 + sy - 2, h - 1)
                    tx_cold = min(cold_x0 + sx + 4, w - 1)
                    ty_cold = min(cold_y0 + sy - 2, h - 1)

                    def put_label(img, text, org, color):
                        x, y = int(org[0]), int(org[1])
                        for dx in (-1, 0, 1):
                            for dy in (-1, 0, 1):
                                if dx or dy:
                                    cv.putText(
                                        img,
                                        text,
                                        (x + dx, y + dy),
                                        cv.FONT_HERSHEY_SIMPLEX,
                                        0.45,
                                        (0, 0, 0),
                                        2,
                                        cv.LINE_AA,
                                    )
                        cv.putText(
                            img,
                            text,
                            (x, y),
                            cv.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            color,
                            1,
                            cv.LINE_AA,
                        )

                    put_label(vis, hot_txt, (tx_hot, ty_hot), (0, 0, 255))
                    put_label(vis, cold_txt, (tx_cold, ty_cold), (255, 255, 0))
            except Exception:
                pass

        qimg = QtGui.QImage(
            vis.data,
            vis.shape[1],
            vis.shape[0],
            vis.strides[0],
            QtGui.QImage.Format_BGR888,
        )
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(qimg))
        self.image_label.setFixedSize(w, h)

        vdd_txt = "--"
        die_txt = "--"
        fc = "--"
        pmin = "--"
        pmax = "--"
        crc = "--"
        ts = "--"
        try:
            if header is not None:
                keys = {str(k).lower(): k for k in header.keys()}
                vdd_key = next(
                    (
                        orig
                        for k, orig in keys.items()
                        if "vdd" in k and "temp" not in k
                    ),
                    None,
                )
                if vdd_key is not None:
                    v = float(header[vdd_key])
                    v = v / 1000.0 if v > 20.0 else v
                    vdd_txt = f"{v:.2f}"
                die_key = next(
                    (
                        orig
                        for k, orig in keys.items()
                        if ("die" in k and "temp" in k)
                        or k in ("die_temp", "die_temperature", "senxor_temperature")
                    ),
                    None,
                )
                if die_key is not None:
                    die_v = float(header[die_key])
                    die_txt = f"{die_v:.2f}"
                fc = header.get("frame_counter", header.get("framecount", "--"))
                pmin = header.get("pixel_min", header.get("min_pixel", "--"))
                pmax = header.get("pixel_max", header.get("max_pixel", "--"))
                crc = header.get("crc", "--")
                ts = header.get("timestamp", header.get("ts", "--"))
        except Exception:
            pass

        try:
            pmin = f"{float(pmin):.2f}"
        except Exception:
            pass
        try:
            pmax = f"{float(pmax):.2f}"
        except Exception:
            pass

        hdr_str = f"frame: {fc} | ts: {ts} | min: {pmin} | max: {pmax} | crc: {crc}"

        tgt_txt = "--"
        if (
            (self.hover_ix is not None)
            and (self.hover_iy is not None)
            and (self.last_frame_float is not None)
        ):
            try:
                tgt_txt = (
                    f"{float(self.last_frame_float[self.hover_iy, self.hover_ix]):.2f}"
                )
            except Exception:
                tgt_txt = "--"

        self.info_die.setText(
            f"VDD: {vdd_txt} V  |  Target: {tgt_txt} °C  |  Die: {die_txt} °C"
        )
        self.info_header.setText(hdr_str)
        self._sync_left_width()  # --- Sensor FPS by wall-time window (robust) ---
        dfc = 1
        try:
            if header is not None:
                fc_raw = header.get("frame_counter", header.get("framecount"))
                if fc_raw is not None:
                    fc_now = (
                        int(str(fc_raw).split()[0], 0)
                        if isinstance(fc_raw, str)
                        else int(fc_raw)
                    )
                    if self._prev_fc is None:
                        self._prev_fc = fc_now
                    dfc = max(0, fc_now - self._prev_fc) or 1
                    self._prev_fc = fc_now
        except Exception:
            pass

        now_wall = time.time()
        if self._sens_win_t0 is None:
            self._sens_win_t0 = now_wall
        self._sens_win_frames += dfc
        dtw = now_wall - self._sens_win_t0
        if dtw >= 0.5:
            inst = self._sens_win_frames / dtw
            if self._sensor_fps_ema is None:
                self._sensor_fps_ema = inst
            else:
                self._sensor_fps_ema = 0.6 * self._sensor_fps_ema + 0.4 * inst
            self.sensor_fps = float(self._sensor_fps_ema)
            self._sens_win_frames = 0
            self._sens_win_t0 = now_wall
        # --- end sensor FPS ---

        self.frame_count += 1
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.current_fps = self.frame_count / (now - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = now
            self.lbl_fps.setText(
                f"FPS: {self.current_fps:.2f} (target: {self.target_fps}, sensor: {self.sensor_fps:.2f})"
            )

        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.current_fps = self.frame_count / (now - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = now
            self.lbl_fps.setText(
                f"FPS: {self.current_fps:.2f} (target: {self.target_fps}, sensor: {self.sensor_fps:.2f})"
            )

    def save_settings(self):
        try:
            data = {
                "palette": self.palette_name,
                "rotate_mode": self.rotate_mode,
                "flip_h": bool(self.chk_flip_h.isChecked()),
                "flip_v": bool(self.chk_flip_v.isChecked()),
                "auto": bool(self.chk_auto.isChecked()),
                "p_low": float(self.spin_p_low.value()),
                "p_high": float(self.spin_p_high.value()),
                "min_manual": float(self.spin_min.value()),
                "max_manual": float(self.spin_max.value()),
                "clahe": bool(self.chk_clahe.isChecked()),
                "clahe_clip": float(self.spin_clip.value()),
                "clahe_tiles_x": int(self.spin_tiles_x.value()),
                "clahe_tiles_y": int(self.spin_tiles_y.value()),
                "bilateral": bool(self.chk_bilat.isChecked()),
                "bilateral_d": int(self.spin_bilat_d.value()),
                "bilateral_sigc": float(self.spin_bilat_sc.value()),
                "bilateral_sigs": float(self.spin_bilat_ss.value()),
                "gauss_sigma": float(self.spin_gauss.value()),
                "unsharp_sigma": float(self.spin_unsharp_sigma.value()),
                "unsharp_amount": float(self.spin_unsharp_amt.value()),
                "scale_factor": int(self.scale_slider.value()),
                "output_dir": str(self.out_dir_edit.text()),
                "target_fps": int(self.spin_target_fps.value()),
                "minmax_markers": bool(self.chk_minmax.isChecked()),
                "minmax_values": bool(self.chk_minmax_vals.isChecked()),
                "win_width": int(self.width()),
                "win_height": int(self.height()),
                "serial_port": (
                    str(self.port_combo.currentText())
                    if self.port_combo.currentText()
                    else ""
                ),
                "auto_connect": bool(self.chk_auto_connect.isChecked()),
                "auto_start": bool(self.chk_auto_start.isChecked()),
            }
            tmp = self.settings_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self.settings_path)
            return True
        except Exception as e:
            try:
                tmp = self.settings_path + ".tmp"
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            print("SAVE ERROR:", e)
            return False

    def load_settings(self):
        try:
            if not os.path.exists(self.settings_path):
                return True
            with open(self.settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print("LOAD ERROR:", e)
            return False

        try:
            self.palette_combo.setCurrentText(
                str(data.get("palette", self.palette_name))
            )
            self.rotate_combo.setCurrentText(
                str(data.get("rotate_mode", self.rotate_mode))
            )
            self.chk_flip_h.setChecked(bool(data.get("flip_h", self.flip_h)))
            self.chk_flip_v.setChecked(bool(data.get("flip_v", self.flip_v)))
            self.chk_auto.setChecked(bool(data.get("auto", True)))
            self.spin_p_low.setValue(float(data.get("p_low", self.p_low)))
            self.spin_p_high.setValue(float(data.get("p_high", self.p_high)))
            self.spin_min.setValue(float(data.get("min_manual", self.min_manual)))
            self.spin_max.setValue(float(data.get("max_manual", self.max_manual)))
            self.chk_clahe.setChecked(bool(data.get("clahe", True)))
            self.spin_clip.setValue(float(data.get("clahe_clip", self.clahe_clip)))
            self.spin_tiles_x.setValue(
                int(data.get("clahe_tiles_x", self.clahe_tiles_x))
            )
            self.spin_tiles_y.setValue(
                int(data.get("clahe_tiles_y", self.clahe_tiles_y))
            )
            self.chk_bilat.setChecked(
                bool(data.get("bilateral", self.filt.use_bilateral))
            )
            self.spin_bilat_d.setValue(
                int(data.get("bilateral_d", self.filt.bilateral_d))
            )
            self.spin_bilat_sc.setValue(
                float(data.get("bilateral_sigc", self.filt.bilateral_sigc))
            )
            self.spin_bilat_ss.setValue(
                float(data.get("bilateral_sigs", self.filt.bilateral_sigs))
            )
            self.spin_gauss.setValue(
                float(data.get("gauss_sigma", self.filt.gauss_sigma))
            )
            self.spin_unsharp_sigma.setValue(
                float(data.get("unsharp_sigma", self.filt.unsharp_sigma))
            )
            self.spin_unsharp_amt.setValue(
                float(data.get("unsharp_amount", self.filt.unsharp_amount))
            )
            self.scale_slider.setValue(int(data.get("scale_factor", self.scale_factor)))
            out_dir = str(data.get("output_dir", self.output_dir))
            if out_dir and os.path.isdir(out_dir):
                self.out_dir_edit.setText(out_dir)
                self.output_dir = out_dir
            val = int(data.get("target_fps", self.target_fps))
            val = max(1, min(25, val))
            self.spin_target_fps.setValue(val)
            self.chk_minmax.setChecked(bool(data.get("minmax_markers", True)))
            self.chk_minmax_vals.setChecked(bool(data.get("minmax_values", True)))
            # Auto flags
            self.chk_auto_connect.setChecked(bool(data.get("auto_connect", True)))
            self.chk_auto_start.setChecked(bool(data.get("auto_start", True)))
            ww = int(data.get("win_width", self.width()))
            hh = int(data.get("win_height", self.height()))
            try:
                self.resize(ww, hh)
            except Exception:
                pass
            sp = str(data.get("serial_port", ""))
            if sp:
                idx = self.port_combo.findText(sp)
                if idx >= 0:
                    self.port_combo.setCurrentIndex(idx)
        except Exception as e:
            print("APPLY SETTINGS ERROR:", e)
            return False
        return True

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        try:
            self._sync_left_width()
            self.mark_dirty()
        except Exception:
            pass
        return super().resizeEvent(event)

    def open_raw_window(self):
        try:
            if self.raw_win is None or not isinstance(self.raw_win, RawWindow):
                self.raw_win = RawWindow(self)
            self.raw_win.show()
            self.raw_win.raise_()
            self.raw_win.activateWindow()
        except Exception as e:
            try:
                self.statusBar().showMessage(f"Raw window error: {e}", 4000)
            except Exception:
                pass

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        try:
            self.save_settings()
        except Exception:
            pass
        return super().closeEvent(event)


class RawWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Raw sensor view (no processing)")
        self.setModal(False)
        self.setMinimumSize(400, 320)
        self.label = QtWidgets.QLabel()
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setStyleSheet("background:black")
        self.scale_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scale_slider.setMinimum(2)
        self.scale_slider.setMaximum(16)
        self.scale_slider.setValue(8)
        self.scale_slider.setToolTip("Scale x (nearest neighbor)")
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.label)
        lay.addWidget(self.scale_slider)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(33)  # ~30 fps update
        self.timer.timeout.connect(self.refresh)
        self.timer.start()

    def refresh(self):
        par = self.parent()
        if par is None:
            return
        arr = getattr(par, "last_raw_u8", None)
        if arr is None:
            return
        # Build grayscale QImage (no palette, raw orientation 80x62)
        h, w = arr.shape[:2]
        qimg = QtGui.QImage(
            arr.data, w, h, arr.strides[0], QtGui.QImage.Format_Grayscale8
        )
        pm = QtGui.QPixmap.fromImage(qimg)
        s = int(self.scale_slider.value())
        pm = pm.scaled(
            w * s, h * s, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.FastTransformation
        )
        self.label.setPixmap(pm)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = Viewer()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
