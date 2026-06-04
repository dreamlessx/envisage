"""Post-processing: CodeFormer face restoration + ArcFace identity gate.

Pipeline order: TPS pre-warp -> depth mod -> FLUX inpainting -> CodeFormer -> ArcFace gate

Ported from preVisage/previsage/postprocess.py with simplifications.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class PostProcessConfig:
    """Post-processing configuration."""

    codeformer_fidelity: float = 0.6  # 0=quality, 1=fidelity
    codeformer_cache_dir: Path = Path.home() / ".cache" / "codeformer"
    identity_threshold: float = 0.6
    max_retries: int = 3


_codeformer_model: dict | None = None


def apply_codeformer(
    image: np.ndarray,
    fidelity: float = 0.6,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """Apply CodeFormer face restoration.

    Args:
        image: BGR uint8 image.
        fidelity: CodeFormer w parameter (0=quality, 1=fidelity).
        cache_dir: Model cache directory.

    Returns:
        Restored BGR uint8 image, or input unchanged on failure.
    """
    global _codeformer_model

    try:
        from codeformer.inference_codeformer import (
            ARCH_REGISTRY,
            FaceRestoreHelper,
            img2tensor,
            load_file_from_url,
            normalize,
            pretrain_model_url,
            tensor2img,
        )
    except ImportError:
        log.warning("CodeFormer not installed (pip install codeformer-pip), skipping")
        return image

    try:
        if _codeformer_model is None:
            net = ARCH_REGISTRY.get("CodeFormer")(
                dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                connect_list=["32", "64", "128", "256"],
            )
            ckpt_path = load_file_from_url(
                url=pretrain_model_url["restoration"],
                model_dir=str(cache_dir or Path.home() / ".cache" / "codeformer"),
                progress=False,
            )
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            net.load_state_dict(checkpoint["params_ema"])
            device = torch.device(DEVICE)
            net.eval().to(device)
            _codeformer_model = {"net": net, "device": device}
            log.info("Initialized CodeFormer on %s", device)

        net = _codeformer_model["net"]
        device = _codeformer_model["device"]

        face_helper = FaceRestoreHelper(
            upscale_factor=1, face_size=512, crop_ratio=(1, 1),
            det_model="retinaface_resnet50", save_ext="png", use_parse=True,
        )
        face_helper.read_image(image)
        face_helper.get_face_landmarks_5(
            only_center_face=False, resize=640, eye_dist_threshold=5
        )
        face_helper.align_warp_face()

        if not face_helper.cropped_faces:
            log.debug("No faces detected for CodeFormer")
            return image

        for cropped_face in face_helper.cropped_faces:
            face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            face_t = normalize(face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            face_t = face_t.unsqueeze(0).to(device)

            with torch.no_grad():
                output = net(face_t, w=fidelity, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))

            face_helper.add_restored_face(restored_face.astype(np.uint8), cropped_face)

        face_helper.get_inverse_affine(None)
        restored = face_helper.paste_faces_to_input_image()

        if restored is None or restored.shape != image.shape:
            log.warning("CodeFormer shape mismatch, returning input")
            return image

        log.info("Applied CodeFormer (w=%.2f) to %d face(s)", fidelity, len(face_helper.cropped_faces))
        return restored.astype(np.uint8)

    except Exception:
        log.exception("CodeFormer failed, returning input")
        return image


def enforce_nasal_symmetry(
    image: np.ndarray,
    landmarks,
    strength: float = 0.4,
) -> np.ndarray:
    """Mirror-average the nose region for bilateral symmetry.

    Extracts the nasal region, flips it horizontally around the nose
    midline, and blends the original with the mirror at the given strength.
    Only affects the nose mask area; rest of the face is untouched.

    Args:
        image: BGR image (H, W, 3).
        landmarks: FaceLandmarks with 478 points.
        strength: 0=no change, 1=full mirror average.

    Returns:
        Image with symmetric nose region.
    """
    from .landmarks import measure_nasal_symmetry, get_region_points

    pts = landmarks.points
    h, w = image.shape[:2]
    sym = measure_nasal_symmetry(landmarks)
    midline_x = sym["midline_x"]

    # Get nose region bounding box with padding
    nose_pts = get_region_points(landmarks, "rhinoplasty")
    if len(nose_pts) < 3:
        return image

    x_min = max(0, int(nose_pts[:, 0].min()) - 10)
    x_max = min(w, int(nose_pts[:, 0].max()) + 10)
    y_min = max(0, int(nose_pts[:, 1].min()) - 10)
    y_max = min(h, int(nose_pts[:, 1].max()) + 10)

    mid_x_local = midline_x - x_min
    patch = image[y_min:y_max, x_min:x_max].copy()
    ph, pw = patch.shape[:2]

    if pw < 4 or ph < 4:
        return image

    # Create mirrored version around nose midline
    mirrored = patch.copy()
    mid_col = int(mid_x_local)
    mid_col = max(1, min(pw - 2, mid_col))

    # Flip left half to right and vice versa
    left_half = patch[:, :mid_col]
    right_half = patch[:, mid_col:]

    lw = left_half.shape[1]
    rw = right_half.shape[1]
    overlap = min(lw, rw)

    if overlap > 2:
        # Average left-flipped and right, right-flipped and left
        left_flipped = left_half[:, ::-1][:, :overlap]
        right_flipped = right_half[:, ::-1][:, :overlap]

        right_region = right_half[:, :overlap]
        left_region = left_half[:, lw - overlap:]

        avg_right = (right_region.astype(np.float32) + left_flipped.astype(np.float32)) / 2
        avg_left = (left_region.astype(np.float32) + right_flipped.astype(np.float32)) / 2

        mirrored[:, mid_col:mid_col + overlap] = avg_right.astype(np.uint8)
        mirrored[:, mid_col - overlap:mid_col] = avg_left.astype(np.uint8)

    # Create soft mask for blending (elliptical, feathered)
    mask = np.zeros((ph, pw), dtype=np.float32)
    cx = int(mid_x_local)
    cy = ph // 2
    rx = pw // 2 - 4
    ry = ph // 2 - 4
    if rx > 0 and ry > 0:
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=8)

    # Blend
    mask_3 = mask[:, :, np.newaxis] * strength
    blended = patch.astype(np.float32) * (1 - mask_3) + mirrored.astype(np.float32) * mask_3

    result = image.copy()
    result[y_min:y_max, x_min:x_max] = np.clip(blended, 0, 255).astype(np.uint8)

    log.info("Nasal symmetry enforced (strength=%.2f, alar_sym=%.3f)",
             strength, sym["alar_symmetry_ratio"])
    return result


def enforce_eye_symmetry(
    image: np.ndarray,
    landmarks,
    strength: float = 0.5,
) -> np.ndarray:
    """Mirror-average UPPER LID FOLD only for bilateral symmetry.

    Only affects the skin fold above the crease; never touches iris,
    sclera, lashes, or lower lid. The upper lid margin (upper eyelid
    edge) is the lower boundary of the averaging region.

    Args:
        image: BGR image (H, W, 3).
        landmarks: FaceLandmarks with 478 points.
        strength: 0=no change, 1=full mirror average.

    Returns:
        Image with symmetric upper lid folds.
    """
    from .landmarks import LEFT_UPPER_LID_FOLD, RIGHT_UPPER_LID_FOLD
    from .landmarks import LEFT_EYE_UPPER, RIGHT_EYE_UPPER

    pts = landmarks.points
    h, w = image.shape[:2]

    # Upper lid fold landmarks ONLY (skin above crease, not the eye opening)
    left_fold_pts = pts[[i for i in LEFT_UPPER_LID_FOLD if i < len(pts)]]
    right_fold_pts = pts[[i for i in RIGHT_UPPER_LID_FOLD if i < len(pts)]]

    # Upper lid margin (the eyelid edge); this is the LOWER boundary
    left_margin_pts = pts[[i for i in LEFT_EYE_UPPER if i < len(pts)]]
    right_margin_pts = pts[[i for i in RIGHT_EYE_UPPER if i < len(pts)]]

    def fold_bbox(fold_pts, margin_pts, pad=8):
        """Bbox from brow to lid margin; excludes eye opening."""
        x_min = max(0, int(fold_pts[:, 0].min()) - pad)
        x_max = min(w, int(fold_pts[:, 0].max()) + pad)
        y_min = max(0, int(fold_pts[:, 1].min()) - pad)  # brow area
        # Lower bound = upper lid margin (NOT lower lid)
        y_max = min(h, int(margin_pts[:, 1].max()) + 2)  # +2px safety
        return x_min, y_min, x_max, y_max

    if len(left_fold_pts) < 3 or len(right_fold_pts) < 3:
        return image

    lx1, ly1, lx2, ly2 = fold_bbox(left_fold_pts, left_margin_pts)
    rx1, ry1, rx2, ry2 = fold_bbox(right_fold_pts, right_margin_pts)

    lh, lw_p = ly2 - ly1, lx2 - lx1
    rh, rw_p = ry2 - ry1, rx2 - rx1
    patch_h = max(lh, rh)
    patch_w = max(lw_p, rw_p)

    if patch_h < 5 or patch_w < 10:
        return image

    left_patch = cv2.resize(image[ly1:ly2, lx1:lx2], (patch_w, patch_h))
    right_patch = cv2.resize(image[ry1:ry2, rx1:rx2], (patch_w, patch_h))

    right_flipped = right_patch[:, ::-1]
    avg_fold = ((left_patch.astype(np.float32) + right_flipped.astype(np.float32)) / 2.0)

    # Soft rectangular mask; full width, feathered at bottom edge (near lid margin)
    mask = np.ones((patch_h, patch_w), dtype=np.float32)
    # Fade out bottom 30% (near the eye opening)
    fade_start = int(patch_h * 0.7)
    for y in range(fade_start, patch_h):
        mask[y, :] *= 1.0 - (y - fade_start) / max(patch_h - fade_start, 1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=3)
    mask_3 = mask[:, :, np.newaxis] * strength

    new_left = left_patch.astype(np.float32) * (1 - mask_3) + avg_fold * mask_3
    avg_fold_flipped = avg_fold[:, ::-1]
    new_right = right_patch.astype(np.float32) * (1 - mask_3[:, ::-1]) + avg_fold_flipped * mask_3[:, ::-1]

    result = image.copy()
    result[ly1:ly2, lx1:lx2] = cv2.resize(
        np.clip(new_left, 0, 255).astype(np.uint8), (lx2 - lx1, ly2 - ly1))
    result[ry1:ry2, rx1:rx2] = cv2.resize(
        np.clip(new_right, 0, 255).astype(np.uint8), (rx2 - rx1, ry2 - ry1))

    log.info("Upper lid symmetry enforced (strength=%.2f, fold only)", strength)
    return result


def arcface_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute ArcFace cosine similarity between two BGR images."""
    try:
        from insightface.app import FaceAnalysis

        if not hasattr(arcface_similarity, "_app"):
            app = FaceAnalysis(
                name="buffalo_l",
                root=str(Path.home() / ".insightface"),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0 if DEVICE == "cuda" else -1, det_size=(640, 640))
            arcface_similarity._app = app

        app = arcface_similarity._app
        f1 = app.get(img1)
        f2 = app.get(img2)

        if not f1 or not f2:
            return float("nan")

        e1, e2 = f1[0].embedding, f2[0].embedding
        return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
    except Exception:
        return float("nan")


def identity_gated_generate(
    generate_fn,
    input_bgr: np.ndarray,
    threshold: float = 0.6,
    max_retries: int = 3,
) -> tuple[np.ndarray, float]:
    """Run generation with ArcFace identity gate.

    Calls generate_fn(seed) up to max_retries times. Each call should
    return a BGR image. Keeps the result with highest ArcFace similarity
    above threshold. If none pass, returns the best attempt.

    Args:
        generate_fn: Callable(seed: int) -> np.ndarray (BGR)
        input_bgr: Original input image for identity comparison.
        threshold: Minimum ArcFace similarity to accept.
        max_retries: Maximum generation attempts.

    Returns:
        (best_result, best_arcface_score)
    """
    best_result = None
    best_score = -1.0

    for attempt in range(max_retries):
        seed = 42 + attempt
        result = generate_fn(seed)

        score = arcface_similarity(input_bgr, result)
        log.info("Identity gate attempt %d: ArcFace=%.3f (threshold=%.2f)",
                 attempt + 1, score, threshold)

        if np.isnan(score):
            if best_result is None:
                best_result = result
            continue

        if score > best_score:
            best_score = score
            best_result = result

        if score >= threshold:
            log.info("Identity gate PASSED at attempt %d", attempt + 1)
            return result, score

    log.warning("Identity gate: best score %.3f after %d attempts (threshold %.2f)",
                best_score, max_retries, threshold)
    return best_result if best_result is not None else input_bgr, best_score


def detect_stubble(
    bgr_image: np.ndarray,
    landmarks: np.ndarray | None = None,
) -> tuple[bool, float]:
    """Detect stubble/facial hair via Laplacian texture variance.

    Ported from preVisage/previsage/clinical.py.

    Args:
        bgr_image: BGR uint8 image.
        landmarks: (478, 2) face landmarks. If None, uses heuristic chin region.

    Returns:
        (detected: bool, confidence: float)
    """
    h, w = bgr_image.shape[:2]
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # Build chin mask
    chin_mask = np.zeros((h, w), dtype=np.uint8)
    if landmarks is not None and len(landmarks) > 152:
        cx, cy = int(landmarks[152][0]), int(landmarks[152][1])
        cv2.ellipse(chin_mask, (cx, cy), (int(w * 0.12), int(h * 0.06)), 0, 0, 360, 255, -1)
    else:
        cx, cy = w // 2, int(h * 0.72)
        cv2.ellipse(chin_mask, (cx, cy), (int(w * 0.12), int(h * 0.06)), 0, 0, 360, 255, -1)

    mask_bool = chin_mask > 0
    if mask_bool.sum() == 0:
        return False, float("nan")

    masked_gray = np.zeros_like(gray)
    masked_gray[mask_bool] = gray[mask_bool]

    laplacian = cv2.Laplacian(masked_gray, cv2.CV_64F)
    lap_values = laplacian[mask_bool]
    if len(lap_values) == 0:
        return False, 0.0

    lap_var = float(np.var(lap_values))
    threshold = 100.0
    confidence = float(np.clip(lap_var / (threshold * 2.0), 0.0, 1.0))
    detected = lap_var > threshold

    if detected:
        log.info("Stubble detected: Laplacian var=%.1f, confidence=%.2f", lap_var, confidence)

    return detected, confidence
