from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RoiSpec:
    x: int
    y: int
    width: int
    height: int


def subtitle_band_roi(
    frame: np.ndarray,
    bottom_fraction: float,
    horizontal_margin_fraction: float,
) -> RoiSpec:
    height, width = frame.shape[:2]
    band_height = max(1, int(height * bottom_fraction))
    x_margin = int(width * horizontal_margin_fraction)
    x = max(0, x_margin)
    y = max(0, height - band_height)
    w = max(1, width - (2 * x_margin))
    h = band_height
    return RoiSpec(x=x, y=y, width=w, height=h)


def crop(frame: np.ndarray, roi: RoiSpec) -> np.ndarray:
    return frame[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]


def preprocess_for_subtitles(cropped: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 5, 50, 50)
    # White subtitles on variable backgrounds benefit from adaptive thresholding.
    thresh = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2,
    )
    return thresh


def calibrate_band(frame: np.ndarray) -> RoiSpec:
    selection = cv2.selectROI(
        "Select Subtitle ROI (press ENTER)",
        frame,
        fromCenter=False,
        showCrosshair=True,
    )
    cv2.destroyWindow("Select Subtitle ROI (press ENTER)")
    x, y, w, h = [int(v) for v in selection]
    if w <= 0 or h <= 0:
        return subtitle_band_roi(frame, bottom_fraction=0.12, horizontal_margin_fraction=0.03)
    return RoiSpec(x=x, y=y, width=w, height=h)

