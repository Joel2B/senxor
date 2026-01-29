import os
import time

import cv2 as cv
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from serial.tools import list_ports

from .backend import connect_senxor, data_to_frame
from .imaging import PALETTES, SRC_H, SRC_W, to_u8_manual, to_u8_percentiles, unsharp_u8


class SensorMixin:
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

        if self.hover_ix is not None and self.hover_iy is not None:
            sx = max(1, int(round(w / self.src_w)))
            sy = max(1, int(round(h / self.src_h)))
            x0 = int(self.hover_ix) * sx
            y0 = int(self.hover_iy) * sy
            cv.rectangle(
                vis, (x0, y0), (x0 + sx - 1, y0 + sy - 1), (0, 255, 0), thickness=-1
            )
            if self.last_frame_float is not None:
                try:
                    val = float(self.last_frame_float[self.hover_iy, self.hover_ix])
                    txt = f"{val:.2f} C"
                    tx = min(x0 + sx + 4, w - 1)
                    ty = min(y0 + sy - 2, h - 1)
                    put_label(vis, txt, (tx, ty), (0, 255, 0))
                except Exception:
                    pass

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
