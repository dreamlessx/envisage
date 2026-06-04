"""Monk Skin Tone Scale classifier for fairness evaluation.

Classifies faces into the 10-point Monk Skin Tone (MST) Scale using
LAB color space analysis of the forehead region. This enables
reporting all evaluation metrics stratified by skin tone.

The MST Scale (Google, 2023) provides a more inclusive alternative to
the Fitzpatrick scale, with 10 tones spanning the full spectrum of
human skin colors.

Reference LAB values for each MST tone are derived from the official
Monk Skin Tone hex colors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Monk Skin Tone Scale reference colors
# ---------------------------------------------------------------------------
# Source: Google Monk Skin Tone Scale
# https://skintone.google/the-scale
# Hex -> RGB -> LAB conversion

MST_HEX = {
    1: "#f6ede4",
    2: "#f3e7db",
    3: "#f7d7c4",
    4: "#eadaba",
    5: "#d7bd96",
    6: "#a07e56",
    7: "#825c43",
    8: "#604134",
    9: "#3a312a",
    10: "#292420",
}


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_lab(r: int, g: int, b: int) -> tuple[float, float, float]:
    """Convert RGB to CIELAB using OpenCV."""
    pixel = np.array([[[b, g, r]]], dtype=np.uint8)  # BGR for OpenCV
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)
    return float(lab[0, 0, 0]), float(lab[0, 0, 1]), float(lab[0, 0, 2])


# Pre-compute LAB values for each MST tone
MST_LAB: dict[int, tuple[float, float, float]] = {}
for _tone, _hex in MST_HEX.items():
    _r, _g, _b = _hex_to_rgb(_hex)
    MST_LAB[_tone] = _rgb_to_lab(_r, _g, _b)


@dataclass
class SkinToneResult:
    """Result of skin tone classification."""

    tone: int  # 1-10 MST scale
    confidence: float  # inverse of distance to nearest reference
    l_value: float  # measured L* (lightness)
    a_value: float  # measured a* (green-red)
    b_value: float  # measured b* (blue-yellow)
    label: str  # human-readable label


# Descriptive labels for each tone range
MST_LABELS = {
    1: "Very Light",
    2: "Light",
    3: "Light-Medium",
    4: "Medium-Light",
    5: "Medium",
    6: "Medium-Dark",
    7: "Dark-Medium",
    8: "Dark",
    9: "Very Dark",
    10: "Deepest",
}

# MediaPipe forehead landmark indices (central forehead region)
FOREHEAD_LANDMARKS = [
    10, 338, 297, 332, 284,  # upper forehead contour
    109, 67, 103, 54, 21,     # upper forehead contour (other side)
    151, 108, 69, 104, 68,    # mid forehead
    337, 299, 333, 298, 301,  # mid forehead (other side)
]


def extract_forehead_region(
    image: np.ndarray,
    landmarks: np.ndarray | None = None,
) -> np.ndarray | None:
    """Extract the forehead region from a face image.

    The forehead is used because it has the most uniform skin tone
    with minimal shadowing from facial features.

    Args:
        image: BGR image.
        landmarks: (478, 2) or (N, 2) face landmarks in pixel coords.
                   If None, uses heuristic forehead crop.

    Returns:
        BGR image of the forehead region, or None if extraction fails.
    """
    h, w = image.shape[:2]

    if landmarks is not None and len(landmarks) > max(FOREHEAD_LANDMARKS):
        # Use landmarks to define forehead polygon
        pts = landmarks[FOREHEAD_LANDMARKS].astype(np.int32)
        hull = cv2.convexHull(pts)

        # Create mask for forehead
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)

        # Erode slightly to avoid hair/eyebrow edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.erode(mask, kernel, iterations=2)

        # Extract pixels
        pixels = image[mask > 0]
        if len(pixels) < 100:
            log.warning("Too few forehead pixels (%d), using heuristic", len(pixels))
            return _heuristic_forehead(image)

        # Return a rectangular crop for visualization
        ys, xs = np.where(mask > 0)
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
        return image[y1:y2, x1:x2]

    return _heuristic_forehead(image)


def _heuristic_forehead(image: np.ndarray) -> np.ndarray:
    """Heuristic forehead crop: upper-center 20% of image."""
    h, w = image.shape[:2]
    y1 = int(h * 0.08)
    y2 = int(h * 0.28)
    x1 = int(w * 0.25)
    x2 = int(w * 0.75)
    return image[y1:y2, x1:x2]


def classify_skin_tone(
    image: np.ndarray,
    landmarks: np.ndarray | None = None,
) -> SkinToneResult:
    """Classify a face image into the 10-point Monk Skin Tone Scale.

    Args:
        image: BGR face image.
        landmarks: Optional (478, 2) face landmarks.

    Returns:
        SkinToneResult with tone (1-10), confidence, and LAB values.
    """
    forehead = extract_forehead_region(image, landmarks)
    if forehead is None or forehead.size == 0:
        log.warning("Could not extract forehead, using full image center")
        h, w = image.shape[:2]
        forehead = image[h // 4 : h // 2, w // 4 : 3 * w // 4]

    # Convert to LAB and compute mean
    lab = cv2.cvtColor(forehead, cv2.COLOR_BGR2LAB)
    mean_l = float(lab[:, :, 0].mean())
    mean_a = float(lab[:, :, 1].mean())
    mean_b = float(lab[:, :, 2].mean())

    # Find nearest MST reference in LAB space (CIE76 distance)
    min_dist = float("inf")
    best_tone = 5

    for tone, (ref_l, ref_a, ref_b) in MST_LAB.items():
        # Weight L* channel more heavily (skin tone is primarily about lightness)
        dist = np.sqrt(
            2.0 * (mean_l - ref_l) ** 2 +
            (mean_a - ref_a) ** 2 +
            (mean_b - ref_b) ** 2
        )
        if dist < min_dist:
            min_dist = dist
            best_tone = tone

    # Confidence: inverse distance, normalized
    confidence = max(0.0, 1.0 - min_dist / 100.0)

    result = SkinToneResult(
        tone=best_tone,
        confidence=confidence,
        l_value=mean_l,
        a_value=mean_a,
        b_value=mean_b,
        label=MST_LABELS[best_tone],
    )
    log.info(
        "Skin tone: MST %d (%s), L*=%.1f a*=%.1f b*=%.1f, conf=%.2f",
        result.tone, result.label, mean_l, mean_a, mean_b, confidence,
    )
    return result


def stratify_by_tone(
    images: list[np.ndarray],
    landmarks_list: list[np.ndarray | None],
) -> dict[int, list[int]]:
    """Classify a batch of images and return indices grouped by MST tone.

    Args:
        images: List of BGR images.
        landmarks_list: Corresponding landmarks (can contain None).

    Returns:
        Dict mapping MST tone (1-10) to list of indices.
    """
    groups: dict[int, list[int]] = {t: [] for t in range(1, 11)}

    for i, (img, lms) in enumerate(zip(images, landmarks_list)):
        lm_arr = lms.points if hasattr(lms, "points") else lms
        result = classify_skin_tone(img, lm_arr)
        groups[result.tone].append(i)

    # Log distribution
    for tone in range(1, 11):
        n = len(groups[tone])
        if n > 0:
            log.info("MST %d (%s): %d samples", tone, MST_LABELS[tone], n)

    return groups


def format_stratified_metrics(
    scores: dict[int, list[float]],
    metric_name: str = "ArcFace",
) -> str:
    """Format metrics stratified by Monk Skin Tone."""
    lines = [
        f"{'MST Tone':<12} {'Label':<15} {'N':<5} {'Mean':<10} {'Std':<10}",
        "-" * 52,
    ]

    all_scores = []
    for tone in range(1, 11):
        vals = scores.get(tone, [])
        if vals:
            mean = np.mean(vals)
            std = np.std(vals)
            all_scores.extend(vals)
            lines.append(
                f"{tone:<12} {MST_LABELS[tone]:<15} {len(vals):<5} "
                f"{mean:<10.4f} {std:<10.4f}"
            )

    lines.append("-" * 52)
    if all_scores:
        lines.append(
            f"{'Overall':<12} {'':<15} {len(all_scores):<5} "
            f"{np.mean(all_scores):<10.4f} {np.std(all_scores):<10.4f}"
        )

    return "\n".join(lines)
