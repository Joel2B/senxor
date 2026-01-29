import json
import os
import sys

from PyQt5 import QtWidgets


class SettingsMixin:
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

    def choose_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose output directory", self.out_dir_edit.text()
        )
        if d:
            self.out_dir_edit.setText(d)
            self.output_dir = d
            self.mark_dirty()

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
