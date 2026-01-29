from PyQt5 import QtCore, QtGui, QtWidgets

from sysmon_helper import attach_sys_monitor

from .imaging import PALETTES
from .raw_window import RawWindow


class UIMixin:
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
        small_css = (
            "font-family: Consolas, 'Courier New', monospace; font-size: 11px; "
            "margin:0px; padding:0px;"
        )
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
