"""Unified measurement extractor for the procedure-fidelity scoring gate.

Given an image and a `measurement_key` (the string stored on each preset
dataclass), returns a single float. NaN on failure. This is the bridge
between the preset system (which declares `measurement_key` per preset)
and the scorer (which needs a single-value delta between input and
candidate).

The keys mapped here correspond to the fields populated by the per-
procedure analyzers in `rhino_config.analyze_rhinoplasty`,
`bleph_config.analyze_blepharoplasty`, and `rhytid_config.analyze_rhytidectomy`,
plus a handful of derived keys used by the fidelity gate.
"""

from __future__ import annotations

import logging

import numpy as np

from .landmarks import (
    NOSE_DORSUM,
    extract_landmarks,
    measure_eyelid_hooding,
    measure_jaw,
    measure_nasal_symmetry,
    measure_nose,
)

log = logging.getLogger(__name__)


def _measure_all(image_bgr: np.ndarray) -> dict[str, float] | None:
    """Extract every measurement the fidelity gate might need. Returns None on failure."""
    try:
        lm = extract_landmarks(image_bgr)
    except Exception as e:
        log.debug("measure_all: landmark extraction failed (%s)", e)
        return None
    if lm is None:
        return None

    out: dict[str, float] = {}

    try:
        nose = measure_nose(lm)
        out["nose_width"] = float(nose["width"])
        out["nose_height"] = float(nose["height"])
        w, _ = lm.image_size
        out["nose_length_ratio"] = float(nose["height"] / max(w, 1))
    except Exception as e:
        log.debug("measure_nose failed: %s", e)

    try:
        sym = measure_nasal_symmetry(lm)
        out["alar_width"] = float(sym["alar_width"])
        out["intercanthal_distance"] = float(sym["intercanthal_distance"])
        out["bridge_width_ratio"] = float(sym["bridge_width_ratio"])
        out["tip_bulbosity"] = float(sym["tip_bulbosity"])
        out["dorsal_deviation_std"] = float(sym["dorsal_deviation_std"])
        # bridge_x_spread: std of NOSE_DORSUM x-coordinates
        bridge_pts = lm.points[[i for i in NOSE_DORSUM if i < len(lm.points)]]
        out["bridge_x_spread"] = (
            float(np.std(bridge_pts[:, 0])) if len(bridge_pts) > 3 else float("nan")
        )
        # tip_droop: tip_y - subnasale_y
        tip = lm.points[1] if 1 < len(lm.points) else lm.points[0]
        subnasale = lm.points[2] if 2 < len(lm.points) else lm.points[0]
        out["tip_droop"] = float(tip[1] - subnasale[1])
    except Exception as e:
        log.debug("measure_nasal_symmetry failed: %s", e)

    try:
        hooding = measure_eyelid_hooding(lm)
        out["left_hooding"] = float(hooding["left_hooding"])
        out["right_hooding"] = float(hooding["right_hooding"])
        out["asymmetry"] = float(hooding["asymmetry"])
        out["hooding_min"] = float(min(hooding["left_hooding"], hooding["right_hooding"]))
    except Exception as e:
        log.debug("measure_eyelid_hooding failed: %s", e)

    try:
        pts = lm.points
        left_lower = pts[145] if 145 < len(pts) else pts[0]
        right_lower = pts[374] if 374 < len(pts) else pts[0]
        left_cheek = pts[116] if 116 < len(pts) else pts[0]
        right_cheek = pts[345] if 345 < len(pts) else pts[0]
        left_bag = float(abs(left_lower[1] - left_cheek[1]))
        right_bag = float(abs(right_lower[1] - right_cheek[1]))
        out["lower_bag"] = float(min(left_bag, right_bag))
    except Exception as e:
        log.debug("lower_bag computation failed: %s", e)

    try:
        jaw = measure_jaw(lm)
        out["jaw_width"] = float(jaw["jaw_width"])
        out["chin_y"] = float(jaw["chin_y"])
        out["jaw_sag"] = float(jaw["chin_y"] - jaw["jaw_mean_y"])
        _, h = lm.image_size
        out["neck_extent_ratio"] = float((h - jaw["chin_y"]) / max(h, 1))
        # marionette_depth: jaw_left_y - mouth_left_y
        pts = lm.points
        mouth_left = pts[61] if 61 < len(pts) else pts[0]
        jaw_left = pts[172] if 172 < len(pts) else pts[0]
        out["marionette_depth"] = float(jaw_left[1] - mouth_left[1])
    except Exception as e:
        log.debug("measure_jaw / marionette failed: %s", e)

    return out


def measure_key(measurement_key: str, image_bgr: np.ndarray) -> float:
    """Return the value of `measurement_key` on `image_bgr`, or NaN on failure.

    This is the canonical entry point for the fidelity gate. If any step
    in the extraction fails (no face detected, measurement error), it
    returns NaN. The scorer treats NaN as "skip this preset check" so a
    candidate is never disqualified on a missing measurement.
    """
    measurements = _measure_all(image_bgr)
    if measurements is None:
        return float("nan")
    return measurements.get(measurement_key, float("nan"))


def measure_all(image_bgr: np.ndarray) -> dict[str, float]:
    """Return every measurement as a dict. Empty dict on failure."""
    result = _measure_all(image_bgr)
    return result if result is not None else {}
