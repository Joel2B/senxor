from PyQt5 import QtCore, QtGui, QtWidgets


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
