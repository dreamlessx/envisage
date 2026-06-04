"""Post-generation validation: "is this okay?" check.

After generating a prediction, validate it before accepting:
1. Identity check: ArcFace similarity to input must be above threshold
2. Outside-mask check: pixels outside mask should be nearly identical to input
3. Artifact check: no extreme color shifts, no face detection failures

If validation fails, the system retries with a more conservative prompt.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


def validate_prediction(
    prediction: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    arcface_score: float,
    identity_threshold: float = 0.5,
    outside_psnr_threshold: float = 25.0,
    color_shift_threshold: float = 30.0,
) -> tuple[bool, list[str]]:
    """Validate a generated prediction. Returns (is_ok, list_of_issues).

    Args:
        prediction: Generated image (RGB, uint8).
        original: Original input image (RGB, uint8).
        mask: Surgical mask (float32 [0,1]).
        arcface_score: ArcFace cosine similarity (pred vs input).
        identity_threshold: Minimum acceptable ArcFace score.
        outside_psnr_threshold: Minimum PSNR outside mask region.
        color_shift_threshold: Maximum mean color shift in outside region.

    Returns:
        (passed, issues): passed=True if all checks pass, issues=list of failures.
    """
    issues = []

    # 1. Identity check
    if arcface_score < identity_threshold and arcface_score > 0:
        issues.append(f"identity_low: ArcFace {arcface_score:.3f} < {identity_threshold}")

    if arcface_score == 0:
        issues.append("face_detection_failed: no face detected in prediction")

    # 2. Outside-mask pixel preservation
    outside_mask = (mask < 0.1).astype(np.float32)
    if outside_mask.sum() > 100:
        pred_f = prediction.astype(np.float32)
        orig_f = original.astype(np.float32)

        if outside_mask.ndim == 2:
            outside_mask_3 = outside_mask[:, :, np.newaxis]
        else:
            outside_mask_3 = outside_mask

        diff = (pred_f - orig_f) * outside_mask_3
        mse = (diff ** 2).sum() / max(outside_mask_3.sum() * 3, 1)
        psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-10))

        if psnr < outside_psnr_threshold:
            issues.append(f"outside_changed: PSNR {psnr:.1f} < {outside_psnr_threshold}")

    # 3. Color shift check (hallucination indicator)
    if outside_mask.sum() > 100:
        mean_pred = (pred_f * outside_mask_3).sum(axis=(0, 1)) / max(outside_mask_3.sum(), 1)
        mean_orig = (orig_f * outside_mask_3).sum(axis=(0, 1)) / max(outside_mask_3.sum(), 1)
        color_shift = np.linalg.norm(mean_pred - mean_orig)

        if color_shift > color_shift_threshold:
            issues.append(f"color_hallucination: shift {color_shift:.1f} > {color_shift_threshold}")

    passed = len(issues) == 0

    if not passed:
        log.warning("Validation FAILED: %s", "; ".join(issues))
    else:
        log.info("Validation passed (ArcFace=%.3f)", arcface_score)

    return passed, issues


def get_conservative_prompt(procedure: str, attempt: int) -> str:
    """Get increasingly conservative prompt for retry attempts.

    Each retry uses a simpler, more constrained prompt.
    """
    if procedure == "rhinoplasty":
        prompts = [
            # Attempt 1: specific
            "a photorealistic frontal portrait of the same person, "
            "natural skin texture, refined nose with straight bridge and defined tip, "
            "clinical photography, high quality",
            # Attempt 2: minimal
            "the same person with a slightly refined nose, natural appearance, "
            "clinical photography",
            # Attempt 3: identity-first
            "the same person, identical appearance, subtly improved nose shape",
        ]
    elif procedure == "blepharoplasty":
        prompts = [
            "a photorealistic frontal portrait of the same person, "
            "refreshed upper eyelids, identical eye color and iris, "
            "studio lighting, high quality",
            "the same person with refreshed eyes, identical features, "
            "clinical photography",
            "the same person, identical appearance, subtly refreshed eyelids",
        ]
    elif procedure == "rhytidectomy":
        prompts = [
            "a photorealistic frontal portrait of the same person, "
            "sharp defined jawline, smooth neck, preserve facial hair, "
            "identical features above jawline, clinical photography",
            "the same person with defined jawline, identical features, "
            "clinical photography",
            "the same person, identical appearance, subtly defined jawline",
        ]
    else:
        prompts = ["the same person, natural appearance, high quality"]

    idx = min(attempt, len(prompts) - 1)
    return prompts[idx]
