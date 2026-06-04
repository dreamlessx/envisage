"""GT-conditioned preset analyzer.

Given a (pre-op, post-op) image pair, recover which of the 24 Envisage
presets were applied by comparing landmark-derived measurements. This
closes the evaluation loop: we can apply the same preset set the
surgeon used without any manual labels.

Output keys are restricted to the trimmed 8-per-procedure taxonomy
defined in `rhino_config.py`, `bleph_config.py`, `rhytid_config.py`.
"""

from __future__ import annotations

import logging

import numpy as np

from .bleph_config import BLEPH_PROCEDURES, BlephAnalysis
from .landmarks import (
    FaceLandmarks,
    measure_eyelid_hooding,
    measure_jaw,
    measure_nasal_symmetry,
    measure_nose,
    NOSE_DORSUM,
)
from .rhino_config import RHINO_PROCEDURES, RhinoAnalysis, Severity
from .rhytid_config import RHYTID_PROCEDURES, RhytidAnalysis

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rhinoplasty (8 presets)
# ---------------------------------------------------------------------------

def detect_rhino_changes(
    pre: FaceLandmarks,
    post: FaceLandmarks,
) -> tuple[list[str], dict[str, int], dict[str, float]]:
    """Detect which rhinoplasty presets were applied between pre and post.

    Returns (active_keys, severity_by_key, measurement_deltas).
    Keys are restricted to the 8 entries in rhino_config.RHINO_PROCEDURES.
    """
    pre_nose = measure_nose(pre)
    post_nose = measure_nose(post)
    pre_sym = measure_nasal_symmetry(pre)
    post_sym = measure_nasal_symmetry(post)

    detected: list[str] = []
    sev: dict[str, int] = {}
    deltas: dict[str, float] = {}

    # #1 dorsal_hump_reduction: bridge-x spread decreases
    pre_bridge = pre.points[[i for i in NOSE_DORSUM if i < len(pre.points)]]
    post_bridge = post.points[[i for i in NOSE_DORSUM if i < len(post.points)]]
    if len(pre_bridge) > 3 and len(post_bridge) > 3:
        pre_spread = float(np.std(pre_bridge[:, 0]))
        post_spread = float(np.std(post_bridge[:, 0]))
        d = post_spread - pre_spread
        deltas["bridge_spread"] = d
        if d < -0.5:
            detected.append("dorsal_hump_reduction")
            sev["dorsal_hump_reduction"] = _severity_by_magnitude(abs(d), [1.0, 2.0, 3.5])

    # #3 dorsal_narrowing: bridge width decreases
    bw_delta = (post_nose["width"] - pre_nose["width"]) / max(pre_nose["width"], 1.0)
    deltas["bridge_width_rel"] = bw_delta
    if bw_delta < -0.03:
        detected.append("dorsal_narrowing")
        sev["dorsal_narrowing"] = _severity_by_magnitude(abs(bw_delta), [0.03, 0.06, 0.10])

    # #4 dorsal_straightening: dorsal deviation std decreases
    dev_delta = post_sym["dorsal_deviation_std"] - pre_sym["dorsal_deviation_std"]
    deltas["dorsal_deviation"] = dev_delta
    if dev_delta < -0.5:
        detected.append("dorsal_straightening")
        sev["dorsal_straightening"] = _severity_by_magnitude(abs(dev_delta), [0.5, 1.5, 3.0])

    # #12 tip_narrowing and #13 tip_definition: tip bulbosity decreases
    bulb_delta = post_sym["tip_bulbosity"] - pre_sym["tip_bulbosity"]
    deltas["tip_bulbosity"] = bulb_delta
    if bulb_delta < -0.03:
        detected.append("tip_narrowing")
        sev["tip_narrowing"] = _severity_by_magnitude(abs(bulb_delta), [0.03, 0.07, 0.12])
    if bulb_delta < -0.02:
        detected.append("tip_definition")
        sev["tip_definition"] = _severity_by_magnitude(abs(bulb_delta), [0.02, 0.05, 0.10])

    # #22 tip_rotation_up: tip y moves up (decreases in image coords)
    pre_tip_y = float(pre.points[1][1]) if 1 < len(pre.points) else 0.0
    post_tip_y = float(post.points[1][1]) if 1 < len(post.points) else 0.0
    tip_shift = post_tip_y - pre_tip_y
    deltas["tip_vertical_shift"] = tip_shift
    if tip_shift < -3.0:
        detected.append("tip_rotation_up")
        sev["tip_rotation_up"] = _severity_by_magnitude(abs(tip_shift), [3.0, 6.0, 10.0])

    # #35 alar_base_narrowing: alar width decreases
    alar_delta = (post_sym["alar_width"] - pre_sym["alar_width"]) / max(pre_sym["alar_width"], 1.0)
    deltas["alar_width_rel"] = alar_delta
    if alar_delta < -0.03:
        detected.append("alar_base_narrowing")
        sev["alar_base_narrowing"] = _severity_by_magnitude(abs(alar_delta), [0.03, 0.06, 0.10])

    # #46 nose_shortening: nose height decreases
    h_delta = (post_nose["height"] - pre_nose["height"]) / max(pre_nose["height"], 1.0)
    deltas["nose_height_rel"] = h_delta
    if h_delta < -0.03:
        detected.append("nose_shortening")
        sev["nose_shortening"] = _severity_by_magnitude(abs(h_delta), [0.03, 0.06, 0.10])

    # Fallback: if nothing detected (frontal-only changes can be subtle), emit
    # the two dominant clinical defaults so downstream prompt-building has
    # something to work with. Flagged via Severity.MILD.
    if not detected:
        for k in ("dorsal_hump_reduction", "tip_definition"):
            detected.append(k)
            sev[k] = Severity.MILD
        deltas["_fallback_applied"] = 1.0

    log.info(
        "GT rhino: %d active (%s), deltas=%s",
        len(detected), ",".join(detected),
        {k: f"{v:.3f}" for k, v in deltas.items()},
    )
    return detected, sev, deltas


# ---------------------------------------------------------------------------
# Blepharoplasty (8 presets)
# ---------------------------------------------------------------------------

def detect_bleph_changes(
    pre: FaceLandmarks,
    post: FaceLandmarks,
) -> tuple[list[str], dict[str, float]]:
    """Detect which blepharoplasty presets were applied."""
    pre_hood = measure_eyelid_hooding(pre)
    post_hood = measure_eyelid_hooding(post)

    detected: list[str] = []
    deltas: dict[str, float] = {}

    # Hooding improvement (hooding ratio INCREASES as hood decreases)
    left_delta = post_hood["left_hooding"] - pre_hood["left_hooding"]
    right_delta = post_hood["right_hooding"] - pre_hood["right_hooding"]
    max_improvement = max(left_delta, right_delta)
    deltas["hooding_left"] = left_delta
    deltas["hooding_right"] = right_delta

    if max_improvement > 0.15:
        detected.append("upper_skin_excision")
        detected.append("upper_dehooding")
    if max_improvement > 0.25:
        detected.append("crease_restoration")
    if max_improvement > 0.20:
        # Fat-pad reduction co-occurs with substantial upper lift
        detected.append("fat_pad_reduction")

    # Symmetry improvement
    asym_delta = pre_hood["asymmetry"] - post_hood["asymmetry"]
    deltas["asymmetry_improvement"] = asym_delta
    if asym_delta > 0.10:
        detected.append("lid_symmetry")

    # Lower-lid family: bag reduction from lid-to-cheek distance delta
    left_bag_pre = _lower_bag(pre)
    left_bag_post = _lower_bag(post)
    bag_delta = left_bag_post - left_bag_pre
    deltas["lower_bag_change"] = bag_delta
    if bag_delta > 1.0:
        detected.append("lower_bag_reduction")
    if bag_delta > 2.0:
        detected.append("tear_trough_smoothing")

    # Crow feet: no landmark signal; co-occur with substantial upper lift
    if max_improvement > 0.30:
        detected.append("crow_feet_softening")

    if not detected:
        detected.append("upper_skin_excision")
        detected.append("upper_dehooding")
        deltas["_fallback_applied"] = 1.0

    log.info("GT bleph: %d active (%s)", len(detected), ",".join(detected))
    return detected, deltas


def _lower_bag(lm: FaceLandmarks) -> float:
    """Mean lid-to-cheek distance (pixels). Higher = more bag/fullness."""
    pts = lm.points
    left_lower = pts[145] if 145 < len(pts) else pts[0]
    right_lower = pts[374] if 374 < len(pts) else pts[0]
    left_cheek = pts[116] if 116 < len(pts) else pts[0]
    right_cheek = pts[345] if 345 < len(pts) else pts[0]
    return float((abs(left_lower[1] - left_cheek[1]) + abs(right_lower[1] - right_cheek[1])) / 2.0)


# ---------------------------------------------------------------------------
# Rhytidectomy (8 presets)
# ---------------------------------------------------------------------------

def detect_rhytid_changes(
    pre: FaceLandmarks,
    post: FaceLandmarks,
) -> tuple[list[str], dict[str, float]]:
    """Detect which rhytidectomy presets were applied."""
    pre_jaw = measure_jaw(pre)
    post_jaw = measure_jaw(post)
    _, h = pre.image_size

    detected: list[str] = []
    deltas: dict[str, float] = {}

    # Jaw sag improvement (sag DECREASES after lift)
    pre_sag = pre_jaw["chin_y"] - pre_jaw["jaw_mean_y"]
    post_sag = post_jaw["chin_y"] - post_jaw["jaw_mean_y"]
    sag_improvement = float(pre_sag - post_sag)
    deltas["jaw_sag_improvement"] = sag_improvement

    # Jawline is always a rhytid goal
    detected.append("jawline_straightening")

    if sag_improvement > h * 0.01:
        detected.append("jowl_elimination")
        detected.append("prejowl_correction")

    # Marionette line depth improvement
    pre_mar = _marionette_depth(pre)
    post_mar = _marionette_depth(post)
    mar_improvement = pre_mar - post_mar
    deltas["marionette_improvement"] = mar_improvement
    if mar_improvement > h * 0.005:
        detected.append("marionette_softening")

    # Nasolabial softening: same signal as jowl work
    if "jowl_elimination" in detected:
        detected.append("nasolabial_softening")

    # Neck family: requires visible neck and is texture-level (not a
    # landmark signal). When jaw work is done, neck work usually co-occurs.
    neck_extent_pre = h - pre_jaw["chin_y"]
    if neck_extent_pre > h * 0.1:
        detected.append("neck_smoothing")
        detected.append("platysmal_band_removal")
        detected.append("submental_definition")

    log.info("GT rhytid: %d active (%s), sag_impr=%.1fpx", len(detected),
             ",".join(detected), sag_improvement)
    return detected, deltas


def _marionette_depth(lm: FaceLandmarks) -> float:
    """Approximate marionette-line depth from mouth-corner to jaw landmarks."""
    pts = lm.points
    mouth_left = pts[61] if 61 < len(pts) else pts[0]
    jaw_left = pts[172] if 172 < len(pts) else pts[0]
    return float(abs(jaw_left[1] - mouth_left[1]))


# ---------------------------------------------------------------------------
# Typed analysis builder + prompt adapter
# ---------------------------------------------------------------------------

def _severity_by_magnitude(value: float, thresholds: tuple[float, float, float] | list[float]) -> int:
    """Map an absolute magnitude to MILD/MODERATE/SEVERE via 3 thresholds."""
    mild, moderate, severe = thresholds
    if value >= severe:
        return Severity.SEVERE
    if value >= moderate:
        return Severity.MODERATE
    if value >= mild:
        return Severity.MILD
    return Severity.NONE


def analyze_gt_pair(
    pre: FaceLandmarks,
    post: FaceLandmarks,
    procedure: str,
):
    """Unified entry point. Returns the procedure-specific Analysis object.

    Callers can then use `.build_prompt(...)` / `.active_keys` / `.summary()`
    just like the inference-time analyzer, keeping the interface uniform
    across pre-op auto-detection and GT-conditioned detection.
    """
    if procedure == "rhinoplasty":
        keys, sev, deltas = detect_rhino_changes(pre, post)
        detected_map = {k: (k in keys) for k in RHINO_PROCEDURES}
        n_active = sum(1 for v in detected_map.values() if v)
        level = 1 if n_active <= 3 else (2 if n_active <= 5 else 3)
        return RhinoAnalysis(
            detected=detected_map,
            severity=sev,
            measurements=deltas,
            level=level,
        )

    if procedure == "blepharoplasty":
        keys, deltas = detect_bleph_changes(pre, post)
        detected_map = {k: (k in keys) for k in BLEPH_PROCEDURES}
        return BlephAnalysis(detected=detected_map, measurements=deltas)

    if procedure == "rhytidectomy":
        keys, deltas = detect_rhytid_changes(pre, post)
        detected_map = {k: (k in keys) for k in RHYTID_PROCEDURES}
        return RhytidAnalysis(detected=detected_map, measurements=deltas)

    raise ValueError(f"Unknown procedure: {procedure!r}")


def build_gt_prompt(procedure: str, detected_changes: list[str]) -> str:
    """Build a prompt from a list of GT-detected preset keys.

    Kept for backward compatibility with earlier callers; prefer
    `analyze_gt_pair(...).build_prompt()` for new code.
    """
    if procedure == "rhinoplasty":
        mapped = {k: (k in detected_changes) for k in RHINO_PROCEDURES}
        return RhinoAnalysis(detected=mapped).build_prompt(max_procedures=3)
    if procedure == "blepharoplasty":
        mapped = {k: (k in detected_changes) for k in BLEPH_PROCEDURES}
        return BlephAnalysis(detected=mapped).build_prompt(max_procedures=3)
    if procedure == "rhytidectomy":
        mapped = {k: (k in detected_changes) for k in RHYTID_PROCEDURES}
        return RhytidAnalysis(detected=mapped).build_prompt(max_procedures=3)
    return ""
