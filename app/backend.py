from typing import TYPE_CHECKING, Any, Optional, Tuple

import importlib.util
import os
import sys
import numpy as np

if TYPE_CHECKING:
    from senxor.mi48 import MI48
else:
    MI48 = Any  # runtime fallback for annotations


def _try_load_utils() -> Tuple[Optional[object], Optional[Exception]]:
    try:
        from senxor import utils as _utils

        return _utils, None
    except Exception as e:
        if getattr(sys, "frozen", False):
            base = getattr(sys, "_MEIPASS", "")
            utils_path = os.path.join(base, "senxor", "utils.py")
            if os.path.exists(utils_path):
                spec = importlib.util.spec_from_file_location(
                    "senxor.utils", utils_path
                )
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    return mod, None
        return None, e


_utils_mod, _utils_err = _try_load_utils()

if _utils_mod is not None:
    data_to_frame = _utils_mod.data_to_frame
    connect_senxor = _utils_mod.connect_senxor
else:

    def data_to_frame(data, shape, hflip=False):
        arr = (
            np.frombuffer(data, dtype=np.float32)
            if isinstance(data, (bytes, bytearray))
            else np.zeros(shape, np.float32)
        )
        return arr.reshape(shape)

    def connect_senxor(src=None, **kwargs):
        msg = "pysenxor not installed"
        if _utils_err is not None:
            msg = f"{msg}: {_utils_err}"
        raise RuntimeError(msg)
