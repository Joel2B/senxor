from dataclasses import dataclass

import cv2 as cv
import numpy as np

SRC_W, SRC_H = 80, 62  # native sensor

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
