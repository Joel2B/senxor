import os
import sys
import time
from typing import List, Optional

import cv2 as cv
import numpy as np
from PyQt5 import QtCore, QtWidgets

from .backend import MI48
from .imaging import FilterParams, SRC_H, SRC_W
from .sensor import SensorMixin
from .settings import SettingsMixin
from .ui import UIMixin


class Viewer(UIMixin, SensorMixin, SettingsMixin, QtWidgets.QMainWindow):
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

        main_mod = sys.modules.get("__main__")
        main_path = getattr(main_mod, "__file__", sys.argv[0])
        base_dir = os.path.dirname(main_path or os.getcwd())
        print(base_dir)
        if os.path.basename(base_dir).lower() == "app":
            base_dir = os.path.dirname(base_dir)
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
