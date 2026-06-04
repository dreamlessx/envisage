"""Clinical degradation augmentation pipeline.

Transforms clean face images to simulate real clinical photography
conditions. Designed as a PyTorch-compatible transforms pipeline.

Augmentations (each applied with independent probability):
  - Harsh directional fluorescent lighting (40%)
  - Color temperature jitter +/-2000K (60%)
  - Green/magenta fluorescent cast (25%)
  - Synthetic surgical pen markings on nose region (35%, input only)
  - JPEG compression artifacts Q=40-95 (30%)
  - Gaussian sensor noise sigma=5-25 (40%)
  - Barrel distortion k1=0.01-0.05 (30%)

Each sample receives 3-5 augmentations randomly selected from the above.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Probabilities and ranges for each augmentation."""

    fluorescent_lighting_prob: float = 0.40
    color_temp_jitter_prob: float = 0.60
    color_temp_range_k: int = 2000
    fluorescent_cast_prob: float = 0.25
    surgical_pen_prob: float = 0.35
    jpeg_compression_prob: float = 0.30
    jpeg_quality_range: tuple[int, int] = (40, 95)
    gaussian_noise_prob: float = 0.40
    gaussian_noise_sigma_range: tuple[float, float] = (5.0, 25.0)
    barrel_distortion_prob: float = 0.30
    barrel_k1_range: tuple[float, float] = (0.01, 0.05)
    min_augments: int = 3
    max_augments: int = 5


# ---------------------------------------------------------------------------
# Individual augmentation functions
# ---------------------------------------------------------------------------

def apply_fluorescent_lighting(
    image: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate harsh directional fluorescent overhead lighting.

    Creates a vertical gradient (bright top, shadowed bottom) with
    slight horizontal asymmetry.
    """
    h, w = image.shape[:2]
    img = image.astype(np.float32)

    # Vertical gradient: bright at top, darker at bottom
    y_grad = np.linspace(1.3, 0.7, h).reshape(-1, 1)
    # Slight horizontal asymmetry
    x_shift = rng.uniform(-0.15, 0.15)
    x_grad = np.linspace(1.0 + x_shift, 1.0 - x_shift, w).reshape(1, -1)

    lighting = y_grad * x_grad
    lighting = lighting[:, :, np.newaxis]

    img = img * lighting
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_color_temp_jitter(
    image: np.ndarray,
    rng: np.random.Generator,
    range_k: int = 2000,
) -> np.ndarray:
    """Shift color temperature by +/-range_k Kelvin.

    Warm (higher K): boost red, reduce blue.
    Cool (lower K): boost blue, reduce red.
    """
    shift = rng.uniform(-range_k, range_k)
    img = image.astype(np.float32)

    # Approximate Kelvin shift as RGB channel scaling
    # Warm: R up, B down. Cool: R down, B up.
    factor = shift / 6500.0  # normalize to ~[-0.3, +0.3]
    img[:, :, 2] *= 1.0 + factor * 0.3  # Red channel (BGR format)
    img[:, :, 0] *= 1.0 - factor * 0.3  # Blue channel

    return np.clip(img, 0, 255).astype(np.uint8)


def apply_fluorescent_cast(
    image: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add green or magenta fluorescent color cast."""
    img = image.astype(np.float32)
    cast_type = rng.choice(["green", "magenta"])

    if cast_type == "green":
        strength = rng.uniform(0.03, 0.10)
        img[:, :, 1] *= 1.0 + strength  # Green channel boost
        img[:, :, 2] *= 1.0 - strength * 0.3  # Slight red reduction
    else:
        strength = rng.uniform(0.03, 0.08)
        img[:, :, 2] *= 1.0 + strength  # Red boost
        img[:, :, 0] *= 1.0 + strength * 0.5  # Blue boost
        img[:, :, 1] *= 1.0 - strength * 0.5  # Green reduction

    return np.clip(img, 0, 255).astype(np.uint8)


def apply_surgical_pen(
    image: np.ndarray,
    rng: np.random.Generator,
    nose_center: tuple[int, int] | None = None,
) -> np.ndarray:
    """Draw synthetic surgical pen markings on the nose region.

    Applied to input images only, never to targets.
    """
    img = image.copy()
    h, w = img.shape[:2]

    if nose_center is None:
        cx, cy = w // 2, int(h * 0.50)
    else:
        cx, cy = nose_center

    # Pen color: blue-purple (typical surgical markers)
    colors = [
        (180, 50, 50),    # blue
        (150, 40, 100),   # purple
        (160, 60, 80),    # blue-purple
    ]
    color = colors[rng.integers(len(colors))]

    # Draw 2-4 curved lines around nose
    num_lines = rng.integers(2, 5)
    for _ in range(num_lines):
        # Generate Bezier-like curve points
        num_pts = rng.integers(4, 8)
        base_x = cx + rng.integers(-int(w * 0.08), int(w * 0.08))
        base_y = cy + rng.integers(-int(h * 0.08), int(h * 0.08))

        pts = []
        for j in range(num_pts):
            px = base_x + rng.integers(-int(w * 0.06), int(w * 0.06))
            py = base_y + int((j - num_pts / 2) * h * 0.03)
            pts.append([px, py])

        pts = np.array(pts, dtype=np.int32)
        thickness = rng.integers(1, 3)
        cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness)

    # Optional: small dots at intersections
    if rng.random() < 0.5:
        for _ in range(rng.integers(2, 5)):
            dx = cx + rng.integers(-int(w * 0.06), int(w * 0.06))
            dy = cy + rng.integers(-int(h * 0.06), int(h * 0.06))
            cv2.circle(img, (dx, dy), rng.integers(1, 3), color, -1)

    return img


def apply_jpeg_compression(
    image: np.ndarray,
    rng: np.random.Generator,
    quality_range: tuple[int, int] = (40, 95),
) -> np.ndarray:
    """Apply JPEG compression artifacts."""
    quality = int(rng.integers(quality_range[0], quality_range[1] + 1))
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    compressed = Image.open(buf)

    return cv2.cvtColor(np.array(compressed), cv2.COLOR_RGB2BGR)


def apply_gaussian_noise(
    image: np.ndarray,
    rng: np.random.Generator,
    sigma_range: tuple[float, float] = (5.0, 25.0),
) -> np.ndarray:
    """Add Gaussian sensor noise."""
    sigma = rng.uniform(sigma_range[0], sigma_range[1])
    noise = rng.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_barrel_distortion(
    image: np.ndarray,
    rng: np.random.Generator,
    k1_range: tuple[float, float] = (0.01, 0.05),
) -> np.ndarray:
    """Apply barrel lens distortion."""
    h, w = image.shape[:2]
    k1 = rng.uniform(k1_range[0], k1_range[1])

    # Camera matrix (centered principal point)
    fx = fy = max(w, h)
    cx_cam, cy_cam = w / 2.0, h / 2.0
    camera_matrix = np.array([
        [fx, 0, cx_cam],
        [0, fy, cy_cam],
        [0, 0, 1],
    ], dtype=np.float64)

    dist_coeffs = np.array([k1, 0, 0, 0, 0], dtype=np.float64)

    # Compute undistortion maps (we want to ADD distortion, so we use
    # initUndistortRectifyMap with the distortion coeffs and then remap)
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, -dist_coeffs, None, camera_matrix, (w, h), cv2.CV_32FC1
    )
    distorted = cv2.remap(image, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return distorted


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

# Registry of augmentations with their config keys
_AUGMENTATIONS = [
    ("fluorescent_lighting", apply_fluorescent_lighting, "fluorescent_lighting_prob"),
    ("color_temp_jitter", None, "color_temp_jitter_prob"),  # special handling
    ("fluorescent_cast", apply_fluorescent_cast, "fluorescent_cast_prob"),
    ("surgical_pen", None, "surgical_pen_prob"),  # special: input only
    ("jpeg_compression", None, "jpeg_compression_prob"),  # special: params
    ("gaussian_noise", None, "gaussian_noise_prob"),  # special: params
    ("barrel_distortion", None, "barrel_distortion_prob"),  # special: params
]


class ClinicalDegradation:
    """PyTorch-compatible transform that degrades images to clinical quality.

    Usage:
        transform = ClinicalDegradation()
        degraded = transform(image_bgr)  # np.ndarray -> np.ndarray

        # For training pairs (pen markings only on input):
        degraded_input = transform(image_bgr, is_input=True)
        degraded_target = transform(image_bgr, is_input=False)
    """

    def __init__(
        self,
        config: AugmentationConfig | None = None,
        seed: int | None = None,
    ):
        self.config = config or AugmentationConfig()
        self.rng = np.random.default_rng(seed)

    def __call__(
        self,
        image: np.ndarray,
        is_input: bool = True,
        nose_center: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Apply random clinical degradation augmentations.

        Args:
            image: BGR uint8 image.
            is_input: If True, surgical pen markings may be applied.
                      If False (target), pen markings are never applied.
            nose_center: (x, y) center for surgical pen placement.

        Returns:
            Degraded BGR uint8 image.
        """
        cfg = self.config

        # Build candidate augmentation list with probabilities
        candidates = []
        candidates.append(("fluorescent_lighting", cfg.fluorescent_lighting_prob))
        candidates.append(("color_temp_jitter", cfg.color_temp_jitter_prob))
        candidates.append(("fluorescent_cast", cfg.fluorescent_cast_prob))
        if is_input:
            candidates.append(("surgical_pen", cfg.surgical_pen_prob))
        candidates.append(("jpeg_compression", cfg.jpeg_compression_prob))
        candidates.append(("gaussian_noise", cfg.gaussian_noise_prob))
        candidates.append(("barrel_distortion", cfg.barrel_distortion_prob))

        # Select 3-5 augmentations
        num_to_apply = self.rng.integers(cfg.min_augments, cfg.max_augments + 1)
        num_to_apply = min(num_to_apply, len(candidates))

        # Weight selection by probability (higher prob = more likely to be picked)
        probs = np.array([p for _, p in candidates])
        probs = probs / probs.sum()
        indices = self.rng.choice(
            len(candidates), size=num_to_apply, replace=False, p=probs
        )
        selected = [candidates[i][0] for i in sorted(indices)]

        # Apply in order
        result = image.copy()
        applied = []

        for name in selected:
            if name == "fluorescent_lighting":
                result = apply_fluorescent_lighting(result, self.rng)
            elif name == "color_temp_jitter":
                result = apply_color_temp_jitter(
                    result, self.rng, cfg.color_temp_range_k
                )
            elif name == "fluorescent_cast":
                result = apply_fluorescent_cast(result, self.rng)
            elif name == "surgical_pen":
                result = apply_surgical_pen(result, self.rng, nose_center)
            elif name == "jpeg_compression":
                result = apply_jpeg_compression(
                    result, self.rng, cfg.jpeg_quality_range
                )
            elif name == "gaussian_noise":
                result = apply_gaussian_noise(
                    result, self.rng, cfg.gaussian_noise_sigma_range
                )
            elif name == "barrel_distortion":
                result = apply_barrel_distortion(
                    result, self.rng, cfg.barrel_k1_range
                )
            applied.append(name)

        log.debug("Applied augmentations: %s", applied)
        return result


def create_test_grid(
    image: np.ndarray,
    num_samples: int = 8,
    seed: int = 42,
) -> np.ndarray:
    """Create a grid of augmented samples for visual inspection.

    Args:
        image: BGR input image.
        num_samples: Number of augmented variants.
        seed: Random seed for reproducibility.

    Returns:
        BGR grid image.
    """
    transform = ClinicalDegradation(seed=seed)

    samples = [image]  # original in top-left
    for i in range(num_samples):
        transform.rng = np.random.default_rng(seed + i + 1)
        augmented = transform(image, is_input=True)
        samples.append(augmented)

    # Arrange in grid
    cols = 3
    rows = (len(samples) + cols - 1) // cols

    # Resize all to same size
    target_h = 256
    resized = []
    for s in samples:
        ratio = target_h / s.shape[0]
        new_w = int(s.shape[1] * ratio)
        resized.append(cv2.resize(s, (new_w, target_h)))

    # Pad to same width
    max_w = max(r.shape[1] for r in resized)
    padded = []
    for r in resized:
        if r.shape[1] < max_w:
            pad = np.full((target_h, max_w - r.shape[1], 3), 255, dtype=np.uint8)
            r = np.hstack([r, pad])
        padded.append(r)

    # Build grid
    grid_rows = []
    for row_idx in range(rows):
        start = row_idx * cols
        end = min(start + cols, len(padded))
        row_imgs = padded[start:end]
        # Pad row if needed
        while len(row_imgs) < cols:
            row_imgs.append(np.full_like(padded[0], 255))
        grid_rows.append(np.hstack(row_imgs))

    grid = np.vstack(grid_rows)

    # Add labels
    labels = ["Original"] + [f"Aug {i+1}" for i in range(num_samples)]
    for i, label in enumerate(labels):
        row_idx = i // cols
        col_idx = i % cols
        x = col_idx * max_w + 5
        y = row_idx * target_h + 20
        cv2.putText(
            grid, label, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
        )

    return grid


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Test clinical augmentation")
    parser.add_argument("input", type=Path, help="Input face image")
    parser.add_argument(
        "--output", type=Path, default=Path("augmentation_test_grid.png"),
        help="Output grid image",
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    img = cv2.imread(str(args.input))
    if img is None:
        raise FileNotFoundError(f"Could not read: {args.input}")

    grid = create_test_grid(img, num_samples=args.num_samples, seed=args.seed)
    cv2.imwrite(str(args.output), grid)
    print(f"Saved augmentation grid ({args.num_samples} samples) to {args.output}")
