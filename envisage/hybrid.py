"""Hybrid TPS + depth modification + inpainting pipeline.

Three-stage surgical simulation:
  1. TPS warp: geometric deformation (narrowing, lifting)
  2. Depth modification: profile change (bridge flattening)
  3. FLUX inpainting: photorealistic texture over the deformed geometry

Each stage handles what it does best:
  - TPS: precise geometric control (alar narrowing, eyelid lift)
  - Depth ControlNet: profile/depth guidance (dorsal hump reduction)
  - FLUX: realistic skin texture, lighting consistency, identity preservation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .landmarks import (
    FaceLandmarks,
    NOSE_ALL,
    NOSE_WINGS,
    LEFT_UPPER_LID_FOLD,
    RIGHT_UPPER_LID_FOLD,
)

log = logging.getLogger(__name__)


@dataclass
class RhinoplastyWarpParams:
    """TPS warp parameters for rhinoplasty (v4).

    Four groups of landmarks are displaced:
      1. Bridge sidewall: thins the nasal bridge (horizontal)
      2. Tip lobule: narrows the tip (horizontal)
      3. Tip center: subtle upward rotation (vertical)
      4. Alar base: narrows nostrils (horizontal)
    All displacements scale with measured nose width.
    """

    # Bridge sidewall landmarks
    left_bridge_indices: list[int] = None
    right_bridge_indices: list[int] = None
    bridge_inward_px: float = 1.0  # minimal; hump removal widens the bridge

    # Tip lobule landmarks
    left_tip_indices: list[int] = None
    right_tip_indices: list[int] = None
    tip_narrow_px: float = 2.5  # moderate tip narrowing; avoid indent artifacts

    # Tip center (upward rotation)
    tip_center_indices: list[int] = None
    tip_up_px: float = 1.0  # minimal rotation; avoid over-angling

    # Alar base landmarks
    left_alar_indices: list[int] = None
    right_alar_indices: list[int] = None
    alar_narrow_px: float = 1.0  # minimal; only for truly wide alae

    def __post_init__(self):
        if self.left_bridge_indices is None:
            self.left_bridge_indices = [193, 245, 188, 174, 217, 126, 142]
        if self.right_bridge_indices is None:
            self.right_bridge_indices = [437, 399, 465, 412, 351, 355, 371]
        if self.left_tip_indices is None:
            self.left_tip_indices = [94, 141, 238]
        if self.right_tip_indices is None:
            self.right_tip_indices = [326, 370, 458]
        if self.tip_center_indices is None:
            self.tip_center_indices = [1, 2]
        if self.left_alar_indices is None:
            self.left_alar_indices = [48, 64, 98, 97, 209]
        if self.right_alar_indices is None:
            self.right_alar_indices = [278, 294, 327, 326, 429]


@dataclass
class BlepharoplastyWarpParams:
    """TPS warp parameters for blepharoplasty."""

    # How much to lift upper eyelid crease (pixels)
    lid_lift_px: float = 5.0  # aggressive lift to expose tarsal platform


def compute_tps_warp(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    image: np.ndarray,
) -> np.ndarray:
    """Apply thin-plate spline warp using scipy RBF interpolation.

    Uses scipy's RBFInterpolator to compute a dense displacement field,
    then cv2.remap to apply it. More reliable than OpenCV's TPS API.

    Args:
        src_pts: (N, 2) source control points (where pixels are now).
        dst_pts: (N, 2) destination control points (where they should go).
        image: BGR image to warp.

    Returns:
        Warped image.
    """
    from scipy.interpolate import RBFInterpolator

    h, w = image.shape[:2]

    # Displacement vectors at control points
    displacements = dst_pts - src_pts  # (N, 2): dx, dy

    # Build RBF interpolators for x and y displacement
    rbf_dx = RBFInterpolator(
        src_pts, displacements[:, 0],
        kernel="thin_plate_spline", smoothing=1.0,
    )
    rbf_dy = RBFInterpolator(
        src_pts, displacements[:, 1],
        kernel="thin_plate_spline", smoothing=1.0,
    )

    # Create grid of all pixel coordinates
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float64)

    # Compute displacement at every pixel
    dx = rbf_dx(grid_pts).reshape(h, w).astype(np.float32)
    dy = rbf_dy(grid_pts).reshape(h, w).astype(np.float32)

    # Build remap coordinates (inverse warp: for each output pixel,
    # where does it come from in the input?)
    # Forward warp: dst = src + displacement
    # Inverse: src = dst - displacement
    map_x = (grid_x - dx).astype(np.float32)
    map_y = (grid_y - dy).astype(np.float32)

    warped = cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return warped


def add_boundary_anchors(
    src: np.ndarray,
    dst: np.ndarray,
    w: int,
    h: int,
    n_per_edge: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Add identity control points along image borders (no duplicates)."""
    border_set = set()
    border = []
    for i in range(n_per_edge):
        t = i / max(n_per_edge - 1, 1)
        pts = [
            (round(t * (w - 1)), 0),
            (round(t * (w - 1)), h - 1),
            (0, round(t * (h - 1))),
            (w - 1, round(t * (h - 1))),
        ]
        for p in pts:
            if p not in border_set:
                border_set.add(p)
                border.append([float(p[0]), float(p[1])])
    border = np.array(border, dtype=np.float32)
    return (
        np.vstack([src, border]),
        np.vstack([dst, border]),
    )


def rhinoplasty_tps_warp(
    image: np.ndarray,
    landmarks: FaceLandmarks,
    params: RhinoplastyWarpParams | None = None,
) -> np.ndarray:
    """Apply TPS warp for rhinoplasty (v4).

    Four displacement groups, all adaptive to measured nose width:
      1. Bridge sidewalls move inward (horizontal)
      2. Tip lobule narrows (horizontal)
      3. Tip center rotates upward (vertical)
      4. Alar base narrows (horizontal)

    Args:
        image: BGR input image.
        landmarks: 478-point face landmarks.
        params: Warp parameters.

    Returns:
        TPS-warped BGR image.
    """
    from .landmarks import measure_nose

    if params is None:
        params = RhinoplastyWarpParams()

    pts = landmarks.points
    h, w = image.shape[:2]

    # Adaptive scale: wider noses get proportionally more displacement
    nose = measure_nose(landmarks)
    scale = max(0.7, min(1.2, nose["width"] / 80.0))  # capped at 1.2 to avoid indent artifacts

    src_list = []
    dst_list = []

    # 0. Dorsal straightening: push bridge landmarks toward midline
    from .landmarks import measure_nasal_symmetry, NOSE_DORSUM
    sym = measure_nasal_symmetry(landmarks)
    midline_x = sym["midline_x"]
    if sym["dorsal_deviation_std"] > 1.5:
        for idx in NOSE_DORSUM:
            if idx >= len(pts):
                continue
            src_list.append(pts[idx].copy())
            d = pts[idx].copy()
            deviation = d[0] - midline_x
            d[0] -= deviation * 0.5 * scale  # push 50% toward midline
            dst_list.append(d)

    # 1. Bridge thinning
    bpx = params.bridge_inward_px * scale
    for idx in params.left_bridge_indices:
        if idx >= len(pts):
            continue
        src_list.append(pts[idx].copy())
        d = pts[idx].copy()
        d[0] += bpx
        dst_list.append(d)

    for idx in params.right_bridge_indices:
        if idx >= len(pts):
            continue
        src_list.append(pts[idx].copy())
        d = pts[idx].copy()
        d[0] -= bpx
        dst_list.append(d)

    # 2. Tip narrowing
    tpx = params.tip_narrow_px * scale
    for idx in params.left_tip_indices:
        if idx >= len(pts):
            continue
        src_list.append(pts[idx].copy())
        d = pts[idx].copy()
        d[0] += tpx
        dst_list.append(d)

    for idx in params.right_tip_indices:
        if idx >= len(pts):
            continue
        src_list.append(pts[idx].copy())
        d = pts[idx].copy()
        d[0] -= tpx
        dst_list.append(d)

    # 3. Tip rotation (upward)
    tup = params.tip_up_px * scale
    for idx in params.tip_center_indices:
        if idx >= len(pts):
            continue
        src_list.append(pts[idx].copy())
        d = pts[idx].copy()
        d[1] -= tup
        dst_list.append(d)

    # 4. Alar base narrowing
    apx = params.alar_narrow_px * scale
    for idx in params.left_alar_indices:
        if idx >= len(pts):
            continue
        src_list.append(pts[idx].copy())
        d = pts[idx].copy()
        d[0] += apx
        dst_list.append(d)

    for idx in params.right_alar_indices:
        if idx >= len(pts):
            continue
        src_list.append(pts[idx].copy())
        d = pts[idx].copy()
        d[0] -= apx
        dst_list.append(d)

    if not src_list:
        log.warning("No rhinoplasty warp points found")
        return image.copy()

    src = np.array(src_list, dtype=np.float32)
    dst = np.array(dst_list, dtype=np.float32)

    src, dst = add_boundary_anchors(src, dst, w, h)

    warped = compute_tps_warp(src, dst, image)
    log.info(
        "Rhinoplasty TPS v4: %d points, bridge=%.1f tip=%.1f tip_up=%.1f alar=%.1f (scale=%.2f)",
        len(src_list), bpx, tpx, tup, apx, scale,
    )
    return warped


def blepharoplasty_tps_warp(
    image: np.ndarray,
    landmarks: FaceLandmarks,
    params: BlepharoplastyWarpParams | None = None,
) -> np.ndarray:
    """Apply adaptive TPS warp for blepharoplasty.

    Adaptive lift per eye based on hooding severity. More hooded = more lift.
    Eye opening landmarks (iris, lash line) are anchored in place.
    """
    from .landmarks import (
        measure_eyelid_hooding,
        LEFT_EYE_UPPER, LEFT_EYE_LOWER,
        RIGHT_EYE_UPPER, RIGHT_EYE_LOWER,
    )

    if params is None:
        params = BlepharoplastyWarpParams()

    pts = landmarks.points
    h, w = image.shape[:2]
    hooding = measure_eyelid_hooding(landmarks)

    src_list = []
    dst_list = []

    # Adaptive lift: more hooded eye gets more lift (5-10px range)
    left_hood = hooding["left_hooding"]
    right_hood = hooding["right_hooding"]
    # Lower hooding ratio = more hooded = needs more lift
    left_lift = max(5.0, min(10.0, params.lid_lift_px * (2.0 / max(left_hood, 0.5))))
    right_lift = max(5.0, min(10.0, params.lid_lift_px * (2.0 / max(right_hood, 0.5))))

    # Lift left upper lid fold
    for idx in LEFT_UPPER_LID_FOLD:
        if idx >= len(pts):
            continue
        src_list.append(pts[idx].copy())
        d = pts[idx].copy()
        d[1] -= left_lift
        dst_list.append(d)

    # Lift right upper lid fold
    for idx in RIGHT_UPPER_LID_FOLD:
        if idx >= len(pts):
            continue
        src_list.append(pts[idx].copy())
        d = pts[idx].copy()
        d[1] -= right_lift
        dst_list.append(d)

    # ANCHOR: eye opening landmarks (iris, lash line); zero displacement
    for idx_list in [LEFT_EYE_UPPER, LEFT_EYE_LOWER, RIGHT_EYE_UPPER, RIGHT_EYE_LOWER]:
        for idx in idx_list:
            if idx >= len(pts):
                continue
            src_list.append(pts[idx].copy())
            dst_list.append(pts[idx].copy())

    # ANCHOR: brow landmarks; zero displacement
    for idx in [70, 63, 105, 66, 107, 300, 293, 334, 296, 336]:
        if idx >= len(pts):
            continue
        src_list.append(pts[idx].copy())
        dst_list.append(pts[idx].copy())

    if not src_list:
        log.warning("No blepharoplasty warp points found")
        return image.copy()

    src = np.array(src_list, dtype=np.float32)
    dst = np.array(dst_list, dtype=np.float32)
    src, dst = add_boundary_anchors(src, dst, w, h)

    warped = compute_tps_warp(src, dst, image)
    log.info(
        "Blepharoplasty TPS: L_lift=%.1f R_lift=%.1f (L_hood=%.2f R_hood=%.2f)",
        left_lift, right_lift, left_hood, right_hood,
    )
    return warped


# Dispatch
TPS_WARP_FNS = {
    "rhinoplasty": rhinoplasty_tps_warp,
    "blepharoplasty": blepharoplasty_tps_warp,
}

TPS_WARP_PARAMS = {
    "rhinoplasty": RhinoplastyWarpParams,
    "blepharoplasty": BlepharoplastyWarpParams,
}


def apply_surgical_tps_warp(
    image: np.ndarray,
    landmarks: FaceLandmarks,
    procedure: str,
) -> np.ndarray:
    """Apply procedure-specific TPS warp.

    For procedures without a TPS warp (rhytidectomy, orthognathic),
    returns the image unchanged.
    """
    warp_fn = TPS_WARP_FNS.get(procedure)
    if warp_fn is None:
        log.info("No TPS warp defined for %s, skipping", procedure)
        return image.copy()
    return warp_fn(image, landmarks)
