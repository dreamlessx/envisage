"""Thin-Plate Spline (TPS) augmentation for synthetic training pair generation.

Generates synthetic pre/post-surgery training pairs by applying
anatomically plausible TPS warps to face images. Each warp simulates
a surgical outcome by displacing landmarks in the procedure region.

Pipeline:
  1. Extract landmarks from a face image
  2. Select control points in the procedure region
  3. Apply random but anatomically constrained displacements
  4. Compute TPS warp and apply to the image
  5. Output: (original, warped) pair for inpainting training

The warps are designed to be subtle and realistic:
  - Rhinoplasty: bridge flattening, tip refinement, wing narrowing
  - Blepharoplasty: lid crease deepening, ptosis correction
  - Orthognathic: jaw advancement/setback, chin projection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .landmarks import (
    NOSE_ALL,
    NOSE_DORSUM,
    NOSE_TIP,
    NOSE_WINGS,
    EYELIDS_ALL,
    LEFT_EYELID,
    RIGHT_EYELID,
    JAW_CONTOUR,
    CHIN,
    FaceLandmarks,
)

log = logging.getLogger(__name__)


@dataclass
class TPSConfig:
    """Configuration for TPS augmentation."""

    # Maximum displacement as fraction of face bounding box
    max_displacement: float = 0.03
    # Number of random augmentations per image
    num_augments: int = 5
    # Random seed (None for truly random)
    seed: int | None = None
    # Whether to add identity (fixed) control points at image corners
    add_boundary_points: bool = True
    # Number of boundary control points per edge
    boundary_points_per_edge: int = 4


@dataclass
class RhinoplastyWarpConfig:
    """Anatomically constrained warp parameters for rhinoplasty."""

    # Dorsal hump reduction: inward displacement of bridge landmarks
    bridge_flatten_range: tuple[float, float] = (0.005, 0.025)
    # Tip refinement: slight upward rotation
    tip_refine_range: tuple[float, float] = (0.005, 0.015)
    # Wing narrowing: inward displacement of alar landmarks
    wing_narrow_range: tuple[float, float] = (0.005, 0.02)


@dataclass
class BlepharoplastyWarpConfig:
    """Warp parameters for blepharoplasty."""

    # Lid crease deepening
    crease_deepen_range: tuple[float, float] = (0.003, 0.012)


@dataclass
class OrthognathicWarpConfig:
    """Warp parameters for orthognathic surgery."""

    # Jaw advancement/setback
    jaw_shift_range: tuple[float, float] = (0.005, 0.025)
    # Chin projection change
    chin_project_range: tuple[float, float] = (0.005, 0.02)


# Procedure-specific warp configs
WARP_CONFIGS: dict[str, object] = {
    "rhinoplasty": RhinoplastyWarpConfig(),
    "blepharoplasty": BlepharoplastyWarpConfig(),
    "orthognathic": OrthognathicWarpConfig(),
}


def compute_tps_warp(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    image: np.ndarray,
) -> np.ndarray:
    """Apply thin-plate spline warp to an image.

    Args:
        src_points: (N, 2) source control points.
        dst_points: (N, 2) destination control points.
        image: BGR image to warp.

    Returns:
        Warped image, same size as input.
    """
    h, w = image.shape[:2]

    # OpenCV TPS requires points as (1, N, 2) float32
    src = src_points.reshape(1, -1, 2).astype(np.float32)
    dst = dst_points.reshape(1, -1, 2).astype(np.float32)

    # Create TPS transformer
    tps = cv2.createThinPlateSplineShapeTransformer()
    matches = [cv2.DMatch(i, i, 0) for i in range(len(src_points))]
    tps.estimateTransformation(dst, src, matches)

    # Apply warp
    warped = tps.warpImage(image)

    # TPS can leave black borders; fill with original pixels
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    black_mask = gray == 0
    if black_mask.any():
        warped[black_mask] = image[black_mask]

    return warped


def generate_rhinoplasty_displacements(
    landmarks: FaceLandmarks,
    rng: np.random.Generator,
    config: RhinoplastyWarpConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate anatomically plausible rhinoplasty displacements.

    Returns:
        (src_points, dst_points) control point pairs.
    """
    if config is None:
        config = RhinoplastyWarpConfig()

    w, h = landmarks.image_size
    scale = np.sqrt(w * h)  # normalization factor
    pts = landmarks.points

    src_list = []
    dst_list = []

    # Bridge flattening: push dorsal landmarks inward (reduce depth)
    for idx in NOSE_DORSUM:
        if idx >= len(pts):
            continue
        src_list.append(pts[idx])
        dx = 0.0
        # Push bridge landmarks slightly toward center and down
        dy = rng.uniform(*config.bridge_flatten_range) * scale
        dst_list.append(pts[idx] + np.array([dx, dy], dtype=np.float32))

    # Tip refinement: slight upward displacement
    for idx in NOSE_TIP:
        if idx >= len(pts) or idx in NOSE_DORSUM:
            continue
        src_list.append(pts[idx])
        dy = -rng.uniform(*config.tip_refine_range) * scale
        dx = 0.0
        dst_list.append(pts[idx] + np.array([dx, dy], dtype=np.float32))

    # Wing narrowing: push alar landmarks inward
    center_x = np.mean([pts[i][0] for i in NOSE_TIP if i < len(pts)])
    for idx in NOSE_WINGS:
        if idx >= len(pts):
            continue
        src_list.append(pts[idx])
        direction = 1.0 if pts[idx][0] < center_x else -1.0
        dx = direction * rng.uniform(*config.wing_narrow_range) * scale
        dst_list.append(pts[idx] + np.array([dx, 0.0], dtype=np.float32))

    return np.array(src_list, dtype=np.float32), np.array(dst_list, dtype=np.float32)


def generate_blepharoplasty_displacements(
    landmarks: FaceLandmarks,
    rng: np.random.Generator,
    config: BlepharoplastyWarpConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate blepharoplasty displacements (lid crease deepening)."""
    if config is None:
        config = BlepharoplastyWarpConfig()

    w, h = landmarks.image_size
    scale = np.sqrt(w * h)
    pts = landmarks.points

    src_list = []
    dst_list = []

    # Upper lids: pull upward to deepen crease
    for idx in LEFT_EYELID + RIGHT_EYELID:
        if idx >= len(pts):
            continue
        src_list.append(pts[idx])
        dy = -rng.uniform(*config.crease_deepen_range) * scale
        dst_list.append(pts[idx] + np.array([0.0, dy], dtype=np.float32))

    return np.array(src_list, dtype=np.float32), np.array(dst_list, dtype=np.float32)


def generate_orthognathic_displacements(
    landmarks: FaceLandmarks,
    rng: np.random.Generator,
    config: OrthognathicWarpConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate orthognathic displacements (jaw/chin changes)."""
    if config is None:
        config = OrthognathicWarpConfig()

    w, h = landmarks.image_size
    scale = np.sqrt(w * h)
    pts = landmarks.points

    src_list = []
    dst_list = []

    # Jaw: horizontal shift (advancement/setback)
    jaw_direction = rng.choice([-1.0, 1.0])
    for idx in JAW_CONTOUR:
        if idx >= len(pts):
            continue
        src_list.append(pts[idx])
        dx = jaw_direction * rng.uniform(*config.jaw_shift_range) * scale
        dst_list.append(pts[idx] + np.array([dx, 0.0], dtype=np.float32))

    # Chin: vertical projection change
    chin_direction = rng.choice([-1.0, 1.0])
    for idx in CHIN:
        if idx >= len(pts):
            continue
        src_list.append(pts[idx])
        dy = chin_direction * rng.uniform(*config.chin_project_range) * scale
        dst_list.append(pts[idx] + np.array([0.0, dy], dtype=np.float32))

    return np.array(src_list, dtype=np.float32), np.array(dst_list, dtype=np.float32)


# Dispatch table
_DISPLACEMENT_GENERATORS = {
    "rhinoplasty": generate_rhinoplasty_displacements,
    "blepharoplasty": generate_blepharoplasty_displacements,
    "orthognathic": generate_orthognathic_displacements,
}


def add_boundary_points(
    src: np.ndarray,
    dst: np.ndarray,
    w: int,
    h: int,
    points_per_edge: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Add identity control points along image borders.

    These anchor the TPS warp so it only deforms the interior.
    """
    border = []
    for i in range(points_per_edge):
        t = i / max(points_per_edge - 1, 1)
        border.append([t * w, 0])
        border.append([t * w, h - 1])
        border.append([0, t * h])
        border.append([w - 1, t * h])

    border = np.array(border, dtype=np.float32)
    src_out = np.vstack([src, border])
    dst_out = np.vstack([dst, border])  # identity (no displacement at borders)
    return src_out, dst_out


def generate_training_pair(
    image: np.ndarray,
    landmarks: FaceLandmarks,
    procedure: str,
    config: TPSConfig | None = None,
    warp_config: object | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a single (original, warped) training pair.

    Args:
        image: BGR input image.
        landmarks: Face landmarks.
        procedure: Surgical procedure name.
        config: TPS augmentation config.
        warp_config: Procedure-specific warp parameters.

    Returns:
        (original, warped) image pair.
    """
    if config is None:
        config = TPSConfig()

    rng = np.random.default_rng(config.seed)

    gen_fn = _DISPLACEMENT_GENERATORS.get(procedure)
    if gen_fn is None:
        raise ValueError(
            f"Unknown procedure: {procedure}. "
            f"Choose from: {list(_DISPLACEMENT_GENERATORS.keys())}"
        )

    src, dst = gen_fn(landmarks, rng, warp_config)

    if len(src) < 3:
        log.warning("Too few control points (%d), skipping warp", len(src))
        return image, image.copy()

    # Add boundary anchors
    if config.add_boundary_points:
        w, h = landmarks.image_size
        src, dst = add_boundary_points(
            src, dst, w, h, config.boundary_points_per_edge
        )

    warped = compute_tps_warp(src, dst, image)
    return image, warped


def generate_training_pairs(
    image: np.ndarray,
    landmarks: FaceLandmarks,
    procedure: str,
    config: TPSConfig | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate multiple (original, warped) training pairs.

    Args:
        image: BGR input image.
        landmarks: Face landmarks.
        procedure: Surgical procedure name.
        config: TPS augmentation config.

    Returns:
        List of (original, warped) pairs.
    """
    if config is None:
        config = TPSConfig()

    pairs = []
    base_seed = config.seed if config.seed is not None else 0

    for i in range(config.num_augments):
        pair_config = TPSConfig(
            max_displacement=config.max_displacement,
            num_augments=1,
            seed=base_seed + i if config.seed is not None else None,
            add_boundary_points=config.add_boundary_points,
            boundary_points_per_edge=config.boundary_points_per_edge,
        )
        pair = generate_training_pair(image, landmarks, procedure, pair_config)
        pairs.append(pair)

    log.info("Generated %d TPS training pairs for %s", len(pairs), procedure)
    return pairs


def save_training_pair(
    original: np.ndarray,
    warped: np.ndarray,
    output_dir: Path,
    prefix: str = "pair",
    index: int = 0,
) -> tuple[Path, Path]:
    """Save a training pair to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    orig_path = output_dir / f"{prefix}_{index:04d}_original.png"
    warp_path = output_dir / f"{prefix}_{index:04d}_warped.png"

    cv2.imwrite(str(orig_path), original)
    cv2.imwrite(str(warp_path), warped)

    return orig_path, warp_path
