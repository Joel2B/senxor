# sysmon_helper.py
# Lightweight helper to attach a CPU/RAM status label + timer to your existing Viewer without refactors.
from PyQt5 import QtCore, QtWidgets
try:
    import psutil
except Exception:
    psutil = None

def attach_sys_monitor(viewer: QtWidgets.QMainWindow):
    """Attach 'CPU: x% | RAM: y MB' to status bar and refresh it every 1.5s.
    Call this once at the end of your _build_ui() or in __init__ after UI is ready.
    """
    try:
        sb = viewer.statusBar()
        lbl = QtWidgets.QLabel("CPU: --% | RAM: -- MB")
        sb.addPermanentWidget(lbl)
        viewer._sysmon_lbl = lbl  # keep reference
    except Exception:
        viewer._sysmon_lbl = None

    if psutil is None:
        viewer._sysmon_proc = None
        return

    try:
        viewer._sysmon_proc = psutil.Process()
        try:
            viewer._sysmon_proc.cpu_percent(None)  # prime
        except Exception:
            pass
    except Exception:
        viewer._sysmon_proc = None

    def _tick():
        try:
            if viewer._sysmon_proc is None or viewer._sysmon_lbl is None:
                return
            mem_mb = viewer._sysmon_proc.memory_info().rss / (1024*1024.0)
            cpu = viewer._sysmon_proc.cpu_percent(None)
            cpu_count = psutil.cpu_count(logical=True) or 1
            cpu = cpu / cpu_count
            viewer._sysmon_lbl.setText(
                f"CPU (app): {cpu:.1f}% | RAM: {mem_mb:.1f} MB"
            )
        except Exception:
            pass

    try:
        t = QtCore.QTimer(viewer)
        t.setInterval(1000)
        t.timeout.connect(_tick)
        t.start()
        viewer._sysmon_timer = t
    except Exception:
        pass
