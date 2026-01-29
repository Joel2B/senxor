import sys

from PyQt5 import QtWidgets

from app.viewer import Viewer


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = Viewer()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
