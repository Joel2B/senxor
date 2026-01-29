from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from senxor.mi48 import MI48
else:
    MI48 = Any  # runtime fallback for annotations

try:
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
