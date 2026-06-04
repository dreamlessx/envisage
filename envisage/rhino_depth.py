"""Rhinoplasty depth map modification module.

Two operations run in sequence:
  1. Bridge straightening: linearize the dorsal ridge depth profile
  2. Nostril reshaping: tip rotation, symmetry correction, alar rim sculpting

Both slot in before ControlNet conditioning to guide FLUX toward
the ideal nose shape.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .landmarks import (
    FaceLandmarks,
    NOSE_DORSUM,
    NOSE_TIP,
    NOSE_WINGS,
    measure_nose,
)

log = logging.getLogger(__name__)


@dataclass
class RhinoplastyDepthParams:
    """Parameters for rhinoplasty depth modification."""

    # Bridge straightening
    bridge_straightness: float = 0.8  # 0=original, 1=perfectly straight
    falloff_width: int = 10  # pixels for lateral Gaussian falloff
    tip_offset: float = 0.0  # raise(+) or lower(-) tip depth anchor

    # Nostril reshaping
    tip_rotation: float = 0.4  # 0=no change, 1=max upward rotation
    symmetry_strength: float = 0.7  # 0=keep asymmetry, 1=full mirror
    alar_rim_smoothing: float = 0.5  # rim sculpting intensity
    alar_width_adjust: float = 0.0  # -1=narrower, +1=wider, 0=unchanged
    nostril_falloff: int = 8  # pixels for nostril blending


def _get_bridge_landmarks(pts: np.ndarray) -> list[int]:
    """Get dorsal ridge landmark indices from nasion to pronasale."""
    # Ordered top (nasion) to bottom (tip)
    bridge = [6, 168, 197, 195, 5, 4]
    return [i for i in bridge if i < len(pts)]


def _get_nostril_landmarks(pts: np.ndarray) -> dict:
    """Get nasal base landmark positions."""
    result = {}
    # Left alar rim
    for name, idx in [
        ("left_alar", 48), ("right_alar", 278),
        ("subnasale", 2), ("pronasale", 1),
        ("left_sill", 209), ("right_sill", 429),
        ("columella", 19), ("tip", 4),
    ]:
        if idx < len(pts):
            result[name] = pts[idx].copy()
    return result


def straighten_bridge(
    depth: np.ndarray,
    landmarks: FaceLandmarks,
    straightness: float = 0.8,
    falloff_width: int = 10,
    tip_offset: float = 0.0,
) -> np.ndarray:
    """Straighten the nasal bridge depth profile.

    Samples depth along the dorsal ridge centerline, fits a linear
    interpolation between nasion and tip depths, then blends the
    straightened profile back with Gaussian lateral falloff.

    Args:
        depth: (H, W) float32 depth map.
        landmarks: Face landmarks.
        straightness: 0=original, 1=perfectly straight.
        falloff_width: Lateral Gaussian blend width in pixels.
        tip_offset: Adjust tip depth anchor (+= raise, -= lower).

    Returns:
        Modified depth map.
    """
    if straightness <= 0:
        return depth.copy()

    pts = landmarks.points
    h, w = depth.shape[:2]
    modified = depth.copy()

    bridge_indices = _get_bridge_landmarks(pts)
    if len(bridge_indices) < 3:
        return modified

    # Sample depth along the dorsal ridge centerline
    bridge_pts = np.array([pts[i] for i in bridge_indices])
    nasion = bridge_pts[0]  # top of bridge
    tip = bridge_pts[-1]  # pronasale

    # Get nasion and tip depth values
    nasion_x, nasion_y = int(nasion[0]), int(nasion[1])
    tip_x, tip_y = int(tip[0]), int(tip[1])

    nasion_x = np.clip(nasion_x, 0, w - 1)
    nasion_y = np.clip(nasion_y, 0, h - 1)
    tip_x = np.clip(tip_x, 0, w - 1)
    tip_y = np.clip(tip_y, 0, h - 1)

    nasion_depth = depth[nasion_y, nasion_x]
    tip_depth = depth[tip_y, tip_x] + tip_offset

    # For each row between nasion and tip, compute the ideal (straight) depth
    y_start = max(0, nasion_y - 5)
    y_end = min(h, tip_y + 5)

    for y in range(y_start, y_end):
        # Parametric position along bridge (0=nasion, 1=tip)
        if tip_y == nasion_y:
            t = 0.5
        else:
            t = (y - nasion_y) / (tip_y - nasion_y)
            t = np.clip(t, 0.0, 1.0)

        # Ideal depth: linear interpolation
        ideal_depth = nasion_depth * (1 - t) + tip_depth * t

        # Centerline x at this y level (interpolated from bridge landmarks)
        cx = nasion[0] * (1 - t) + tip[0] * t
        cx = int(np.clip(cx, 0, w - 1))

        # Apply straightened depth with Gaussian lateral falloff
        for x in range(max(0, cx - falloff_width * 3), min(w, cx + falloff_width * 3)):
            dist = abs(x - cx)
            weight = np.exp(-dist ** 2 / (2 * falloff_width ** 2))
            weight *= straightness

            # Only modify within the bridge region (weight falls off laterally)
            if weight > 0.01:
                original = modified[y, x]
                modified[y, x] = original * (1 - weight) + ideal_depth * weight

    log.info(
        "Bridge straightened: nasion_depth=%.1f, tip_depth=%.1f, straightness=%.2f",
        nasion_depth, tip_depth, straightness,
    )
    return modified


def reshape_nostrils(
    depth: np.ndarray,
    landmarks: FaceLandmarks,
    tip_rotation: float = 0.4,
    symmetry_strength: float = 0.7,
    alar_rim_smoothing: float = 0.5,
    alar_width_adjust: float = 0.0,
    falloff: int = 8,
) -> np.ndarray:
    """Reshape nostrils: tip rotation, symmetry, alar rim sculpting.

    Args:
        depth: (H, W) float32 depth map.
        landmarks: Face landmarks.
        tip_rotation: 0=no change, 1=max upward rotation.
        symmetry_strength: 0=keep asymmetry, 1=full mirror.
        alar_rim_smoothing: Rim sculpting intensity.
        alar_width_adjust: -1=narrower, +1=wider.
        falloff: Pixels for blending.

    Returns:
        Modified depth map.
    """
    pts = landmarks.points
    h, w = depth.shape[:2]
    modified = depth.copy()

    nostril_lms = _get_nostril_landmarks(pts)
    if "pronasale" not in nostril_lms or "subnasale" not in nostril_lms:
        return modified

    pronasale = nostril_lms["pronasale"]
    subnasale = nostril_lms["subnasale"]

    # --- 1. Tip rotation (nostril exposure) ---
    if tip_rotation > 0 and "tip" in nostril_lms:
        tip_pt = nostril_lms["tip"]
        tip_x, tip_y = int(tip_pt[0]), int(tip_pt[1])
        pro_x, pro_y = int(pronasale[0]), int(pronasale[1])
        sub_x, sub_y = int(subnasale[0]), int(subnasale[1])

        # Region: from pronasale down to subnasale + some below
        y_start = max(0, pro_y)
        y_end = min(h, sub_y + int(falloff * 2))

        for y in range(y_start, y_end):
            t = (y - pro_y) / max(sub_y - pro_y, 1)
            t = np.clip(t, 0.0, 1.5)

            # Deepen infratip region to simulate upward rotation
            # Max effect at subnasale level, tapering above and below
            rotation_weight = np.sin(np.clip(t, 0, 1) * np.pi) * tip_rotation
            depth_shift = rotation_weight * 8.0  # max 8 depth units

            # Apply with lateral falloff centered on nose midline
            cx = int(pro_x * (1 - t) + sub_x * t)
            for x in range(max(0, cx - falloff * 3), min(w, cx + falloff * 3)):
                dist = abs(x - cx)
                lateral_w = np.exp(-dist ** 2 / (2 * falloff ** 2))
                if lateral_w > 0.01:
                    modified[y, x] -= depth_shift * lateral_w

        log.info("Tip rotation: %.2f", tip_rotation)

    # --- 2. Nostril symmetry correction ---
    if symmetry_strength > 0 and "left_alar" in nostril_lms and "right_alar" in nostril_lms:
        left_alar = nostril_lms["left_alar"]
        right_alar = nostril_lms["right_alar"]

        # Face midline
        midline_x = (left_alar[0] + right_alar[0]) / 2.0
        alar_y = int((left_alar[1] + right_alar[1]) / 2.0)

        # Extract depth patches around each nostril
        patch_h = int(abs(pronasale[1] - subnasale[1]) * 1.5) + 1
        patch_w = int(abs(right_alar[0] - left_alar[0]) * 0.4) + 1

        ly = max(0, alar_y - patch_h // 2)
        lx = max(0, int(left_alar[0]) - patch_w // 2)
        ry = ly
        rx = max(0, int(right_alar[0]) - patch_w // 2)

        # Ensure patches are within bounds
        ly2 = min(h, ly + patch_h)
        lx2 = min(w, lx + patch_w)
        ry2 = min(h, ry + patch_h)
        rx2 = min(w, rx + patch_w)

        left_patch = modified[ly:ly2, lx:lx2].copy()
        right_patch = modified[ry:ry2, rx:rx2].copy()

        # Make patches same size
        min_h = min(left_patch.shape[0], right_patch.shape[0])
        min_w = min(left_patch.shape[1], right_patch.shape[1])
        if min_h > 0 and min_w > 0:
            left_patch = left_patch[:min_h, :min_w]
            right_patch = right_patch[:min_h, :min_w]

            # Mirror-average: flip right patch and blend with left
            right_flipped = right_patch[:, ::-1]
            avg_patch = (left_patch + right_flipped) / 2.0

            # Blend symmetric version back
            s = symmetry_strength
            modified[ly:ly + min_h, lx:lx + min_w] = (
                modified[ly:ly + min_h, lx:lx + min_w] * (1 - s) + avg_patch * s
            )
            modified[ry:ry + min_h, rx:rx + min_w] = (
                modified[ry:ry + min_h, rx:rx + min_w] * (1 - s) + avg_patch[:, ::-1] * s
            )

        log.info("Nostril symmetry: %.2f", symmetry_strength)

    # --- 3. Alar rim sculpting ---
    if alar_rim_smoothing > 0 and "left_alar" in nostril_lms and "right_alar" in nostril_lms:
        left_alar = nostril_lms["left_alar"]
        right_alar = nostril_lms["right_alar"]

        for alar_pt in [left_alar, right_alar]:
            ax, ay = int(alar_pt[0]), int(alar_pt[1])
            r = falloff * 2

            y1 = max(0, ay - r)
            y2 = min(h, ay + r)
            x1 = max(0, ax - r)
            x2 = min(w, ax + r)

            patch = modified[y1:y2, x1:x2].copy()
            if patch.size == 0:
                continue

            # Smooth the rim area
            smoothed = cv2.GaussianBlur(patch, (0, 0), sigmaX=falloff * 0.7)

            # Blend smoothed version
            for py in range(patch.shape[0]):
                for px in range(patch.shape[1]):
                    dist = np.sqrt((py - r) ** 2 + (px - r) ** 2) / r
                    weight = max(0, 1 - dist) * alar_rim_smoothing
                    patch[py, px] = patch[py, px] * (1 - weight) + smoothed[py, px] * weight

            modified[y1:y2, x1:x2] = patch

        log.info("Alar rim smoothing: %.2f", alar_rim_smoothing)

    # --- 4. Alar width adjustment ---
    if abs(alar_width_adjust) > 0.05 and "left_alar" in nostril_lms and "right_alar" in nostril_lms:
        left_alar = nostril_lms["left_alar"]
        right_alar = nostril_lms["right_alar"]
        midline_x = (left_alar[0] + right_alar[0]) / 2.0

        # Compress/expand depth field laterally around nostrils
        alar_y = int((left_alar[1] + right_alar[1]) / 2.0)
        alar_span = abs(right_alar[0] - left_alar[0])
        region_h = int(alar_span * 0.8)

        y1 = max(0, alar_y - region_h // 2)
        y2 = min(h, alar_y + region_h // 2)
        x1 = max(0, int(midline_x - alar_span))
        x2 = min(w, int(midline_x + alar_span))

        if x2 > x1 and y2 > y1:
            patch = modified[y1:y2, x1:x2].copy()
            ph, pw = patch.shape

            # Scale factor for lateral compression/expansion
            scale = 1.0 - alar_width_adjust * 0.15  # +-15% max

            # Remap x coordinates
            center_local = midline_x - x1
            new_xs = np.arange(pw, dtype=np.float32)
            new_xs = center_local + (new_xs - center_local) * scale

            map_x = np.tile(new_xs, (ph, 1)).astype(np.float32)
            map_y = np.tile(np.arange(ph, dtype=np.float32).reshape(-1, 1), (1, pw))

            remapped = cv2.remap(patch, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            # Blend with falloff at edges
            blend_mask = np.ones((ph, pw), dtype=np.float32)
            fade = min(falloff * 2, pw // 4)
            for i in range(fade):
                blend_mask[:, i] = i / fade
                blend_mask[:, pw - 1 - i] = i / fade
            for i in range(min(fade, ph // 4)):
                blend_mask[i, :] *= i / fade
                blend_mask[ph - 1 - i, :] *= i / fade

            modified[y1:y2, x1:x2] = patch * (1 - blend_mask) + remapped * blend_mask

        log.info("Alar width adjust: %.2f", alar_width_adjust)

    return modified


def apply_rhinoplasty_modifications(
    depth: np.ndarray,
    landmarks: FaceLandmarks,
    params: RhinoplastyDepthParams | None = None,
) -> np.ndarray:
    """Apply full rhinoplasty depth modifications.

    Runs bridge straightening first, then nostril reshaping,
    so nostril edits respect the updated bridge geometry.

    Args:
        depth: (H, W) float32 depth map.
        landmarks: Face landmarks.
        params: Modification parameters.

    Returns:
        Modified depth map with bridge straightened and nostrils reshaped.
    """
    if params is None:
        params = RhinoplastyDepthParams()

    # Step 1: Bridge straightening
    modified = straighten_bridge(
        depth, landmarks,
        straightness=params.bridge_straightness,
        falloff_width=params.falloff_width,
        tip_offset=params.tip_offset,
    )

    # Step 2: Nostril reshaping
    modified = reshape_nostrils(
        modified, landmarks,
        tip_rotation=params.tip_rotation,
        symmetry_strength=params.symmetry_strength,
        alar_rim_smoothing=params.alar_rim_smoothing,
        alar_width_adjust=params.alar_width_adjust,
        falloff=params.nostril_falloff,
    )

    return modified
