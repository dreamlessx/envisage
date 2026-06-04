"""Generalized surgical prediction pipeline.

Unified pipeline for all procedures with:
- Input validation (face detection, size, pose)
- Adaptive parameters based on measured anatomy
- Seed sweep with ArcFace identity gate
- Normalize to 512x512 with padding (not stretch)
- Procedure-aware and anatomy-aware prompts
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from . import masks as masks_mod
from .landmarks import (
    FaceLandmarks,
    extract_landmarks,
    measure_nose,
    measure_eyelid_hooding,
    measure_jaw,
    estimate_head_pose,
)
from .masks import (
    MaskConfig,
    generate_mask,
    generate_adaptive_bleph_mask,
    generate_adaptive_rhytid_mask,
)
from .depth import DepthEstimator, modify_depth
from .hybrid import apply_surgical_tps_warp

log = logging.getLogger(__name__)

SEEDS = [42, 123, 456]


def _active_bleph_mask_config() -> MaskConfig:
    factory = getattr(masks_mod, "active_bleph_mask_config", None)
    if callable(factory):
        return factory()
    return MaskConfig(dilation_px=10, feather_sigma=5)


def _active_rhyt_mask_config() -> MaskConfig:
    factory = getattr(masks_mod, "active_rhyt_mask_config", None)
    if callable(factory):
        return factory()
    return MaskConfig(dilation_px=8, feather_sigma=5)


@dataclass
class ValidationResult:
    """Result of input image validation."""

    valid: bool
    message: str
    face_area_pct: float = 0.0
    yaw_degrees: float = 0.0
    image_size: tuple[int, int] = (0, 0)


@dataclass
class PipelineResult:
    """Result of running the generalized pipeline."""

    prediction: np.ndarray  # BGR uint8
    mask: np.ndarray  # float32 [0, 1]
    depth_original: np.ndarray  # float32
    depth_modified: np.ndarray  # float32
    arcface_score: float
    seed_used: int
    procedure: str
    landmarks: FaceLandmarks | None


def validate_input(
    image: np.ndarray,
    min_face_pct: float = 20.0,
    max_yaw: float = 30.0,
    min_resolution: int = 256,
) -> ValidationResult:
    """Validate input image for pipeline processing.

    Args:
        image: BGR image.
        min_face_pct: Minimum face area as percent of image area.
        max_yaw: Maximum head yaw in degrees.
        min_resolution: Minimum image dimension in pixels.

    Returns:
        ValidationResult with valid flag and message.
    """
    h, w = image.shape[:2]

    if min(h, w) < min_resolution:
        return ValidationResult(
            valid=False,
            message=f"Image too small: {w}x{h} (minimum {min_resolution}px)",
            image_size=(w, h),
        )

    landmarks = extract_landmarks(image)
    if landmarks is None:
        return ValidationResult(
            valid=False,
            message="No face detected in image",
            image_size=(w, h),
        )

    # Check face size
    pts = landmarks.points
    face_x_range = pts[:, 0].max() - pts[:, 0].min()
    face_y_range = pts[:, 1].max() - pts[:, 1].min()
    face_area = face_x_range * face_y_range
    image_area = w * h
    face_pct = 100.0 * face_area / image_area

    if face_pct < min_face_pct:
        return ValidationResult(
            valid=False,
            message=f"Face too small: {face_pct:.0f}% of image (minimum {min_face_pct}%)",
            face_area_pct=face_pct,
            image_size=(w, h),
        )

    # Check yaw
    pose = estimate_head_pose(landmarks)
    yaw = abs(pose["yaw_degrees"])
    if yaw > max_yaw:
        return ValidationResult(
            valid=False,
            message=f"Face in profile: yaw={yaw:.0f} degrees (maximum {max_yaw})",
            face_area_pct=face_pct,
            yaw_degrees=yaw,
            image_size=(w, h),
        )

    return ValidationResult(
        valid=True,
        message="OK",
        face_area_pct=face_pct,
        yaw_degrees=yaw,
        image_size=(w, h),
    )


def normalize_to_square(
    image: np.ndarray,
    target_size: int = 512,
) -> tuple[np.ndarray, dict]:
    """Normalize image to target_size x target_size with padding (not stretch).

    Returns:
        (padded_image, pad_info) where pad_info contains the info needed
        to un-pad back to original aspect ratio.
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Pad to square
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_REFLECT_101,
    )

    pad_info = {
        "original_size": (w, h),
        "scale": scale,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right,
        "new_w": new_w,
        "new_h": new_h,
    }

    return padded, pad_info


def unnormalize_from_square(
    image: np.ndarray,
    pad_info: dict,
) -> np.ndarray:
    """Remove padding and resize back to original dimensions."""
    h, w = image.shape[:2]
    pt = pad_info["pad_top"]
    pl = pad_info["pad_left"]
    new_h = pad_info["new_h"]
    new_w = pad_info["new_w"]

    # Remove padding
    cropped = image[pt:pt + new_h, pl:pl + new_w]

    # Resize back to original
    orig_w, orig_h = pad_info["original_size"]
    return cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)


def build_adaptive_prompt(
    procedure: str,
    landmarks: FaceLandmarks | None = None,
) -> str:
    """Build procedure-aware and anatomy-aware prompt.

    Args:
        procedure: Surgical procedure name.
        landmarks: Face landmarks for anatomy-aware adjustments.

    Returns:
        Text prompt for diffusion model.
    """
    if landmarks is not None and procedure == "rhinoplasty":
        return _build_rhinoplasty_prompt(landmarks)
    if landmarks is not None and procedure == "blepharoplasty":
        return _build_blepharoplasty_prompt(landmarks)
    if procedure == "rhytidectomy":
        return _build_rhytidectomy_prompt(landmarks)

    base_prompts = {
        "rhinoplasty": (
            "a photorealistic frontal portrait of the same person, "
            "natural skin texture with visible pores, refined symmetric nose, "
            "straight narrow bridge, defined nasal tip, "
            "clinical photography lighting, high quality"
        ),
        "blepharoplasty": (
            "a photorealistic frontal portrait of the same person, "
            "natural skin texture, refreshed eyelids with smooth contours, "
            "studio lighting, high quality"
        ),
        "orthognathic": (
            "a photorealistic frontal portrait of the same person, "
            "natural skin texture, corrected jaw alignment, balanced facial proportions, "
            "studio lighting, high quality"
        ),
    }
    return base_prompts.get(procedure, base_prompts["rhinoplasty"])


def _build_rhinoplasty_prompt(landmarks: FaceLandmarks) -> str:
    """Build measurement-aware rhinoplasty prompt using Daniel taxonomy.

    Auto-detects which sub-procedures apply and builds a checkbox-style
    prompt. Based on Rollin K. Daniel, "Mastering Rhinoplasty" (2nd Ed).
    """
    from .rhino_config import analyze_rhinoplasty
    analysis = analyze_rhinoplasty(landmarks)
    return analysis.build_prompt()


def _build_blepharoplasty_prompt(landmarks: FaceLandmarks) -> str:
    """Build measurement-aware blepharoplasty prompt using sub-procedure taxonomy."""
    from .bleph_config import analyze_blepharoplasty
    analysis = analyze_blepharoplasty(landmarks)
    return analysis.build_prompt()


def _build_rhytidectomy_prompt(landmarks: FaceLandmarks | None) -> str:
    """Build rhytidectomy prompt using sub-procedure taxonomy.

    MASK: jawline + neck ONLY. Double-mask compositing for blending.
    """
    if landmarks is not None:
        from .rhytid_config import analyze_rhytidectomy
        analysis = analyze_rhytidectomy(landmarks)
        return analysis.build_prompt()
    return (
        "a photorealistic frontal portrait of the same person, "
        "ruler-straight jawline from ear to chin, no jowling, "
        "smooth neck skin without texture or pores, identical neck size, "
        "preserve all facial hair including stubble and beard, "
        "identical features above the jawline including eyes nose and forehead, "
        "clinical photography lighting, high quality"
    )


def _coerce_active_adapters(transformer) -> tuple[str, ...] | None:
    """Best-effort normalization of the transformer's active adapter state."""
    active = getattr(transformer, "active_adapters", None)
    if active is None:
        return None
    if callable(active):
        active = active()
    if active is None:
        return ()
    if isinstance(active, str):
        return (active,)
    if isinstance(active, dict):
        return tuple(str(name) for name, enabled in active.items() if enabled)
    if isinstance(active, (list, tuple, set)):
        return tuple(str(name) for name in active)
    return (str(active),)


def _ensure_procedure_adapter_active(pipe, procedure: str) -> None:
    """Fail fast if a named-adapter pipeline is not actually set to the procedure."""
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return

    set_adapter = getattr(transformer, "set_adapter", None)
    if not callable(set_adapter):
        return

    try:
        set_adapter(procedure)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to activate adapter {procedure!r} before inference"
        ) from exc

    active = _coerce_active_adapters(transformer)
    if active is None:
        log.info(
            "Requested adapter %s before inference; active adapter state unavailable",
            procedure,
        )
        return
    if procedure not in active:
        raise RuntimeError(
            f"Adapter activation mismatch before inference: requested {procedure!r}, "
            f"active={active!r}"
        )
    log.info("Active adapters before inference for %s: %s", procedure, active)


def run_single_seed(
    pipe,
    has_controlnet: bool,
    input_pil: Image.Image,
    mask: np.ndarray,
    modified_depth: np.ndarray,
    prompt: str,
    procedure: str,
    seed: int,
    num_steps: int = 20,
    target_size: int = 512,
    inpainting_strength: float = 0.75,
    controlnet_scale: float = 0.5,
) -> Image.Image:
    """Run FLUX inpainting for a single seed."""
    size = (target_size, target_size)
    image = input_pil.resize(size, Image.LANCZOS)

    mask_r = cv2.resize(mask, size) if mask.shape[:2] != (target_size, target_size) else mask
    mask_pil = Image.fromarray((mask_r * 255).astype(np.uint8))

    gen_kwargs = {
        "prompt": prompt,
        "image": image,
        "mask_image": mask_pil,
        "height": target_size,
        "width": target_size,
        "strength": inpainting_strength,
        "guidance_scale": 3.5,
        "num_inference_steps": num_steps,
        "generator": torch.Generator(device="cpu").manual_seed(seed),
    }

    if has_controlnet:
        depth_r = cv2.resize(modified_depth, size) if modified_depth.shape[:2] != (target_size, target_size) else modified_depth
        depth_rgb = np.stack([depth_r.astype(np.uint8)] * 3, axis=-1)
        gen_kwargs["control_image"] = Image.fromarray(depth_rgb)
        gen_kwargs["controlnet_conditioning_scale"] = controlnet_scale

    _ensure_procedure_adapter_active(pipe, procedure)
    result = pipe(**gen_kwargs)
    return result.images[0].resize(input_pil.size, Image.LANCZOS)


def compute_arcface_score(img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> float:
    """Compute ArcFace similarity between two BGR images."""
    try:
        from insightface.app import FaceAnalysis

        if not hasattr(compute_arcface_score, "_app"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            app = FaceAnalysis(
                name="buffalo_l",
                root=str(Path.home() / ".insightface"),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                if device == "cuda"
                else ["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(320, 320))
            compute_arcface_score._app = app

        app = compute_arcface_score._app
        f1 = app.get(img1_bgr)
        f2 = app.get(img2_bgr)
        if not f1 or not f2:
            return float("nan")
        e1, e2 = f1[0].embedding, f2[0].embedding
        return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
    except Exception:
        return float("nan")


def run_pipeline(
    pipe,
    has_controlnet: bool,
    input_bgr: np.ndarray,
    procedure: str,
    depth_estimator: DepthEstimator,
    intensity_pct: float = 100.0,
    num_steps: int = 20,
    seed_sweep: bool = True,
    seeds: list[int] | None = None,
    validate: bool = True,
) -> PipelineResult | None:
    """Run the full generalized pipeline.

    Args:
        pipe: FLUX pipeline.
        has_controlnet: Whether ControlNet is available.
        input_bgr: BGR input image.
        procedure: Surgical procedure name.
        depth_estimator: Depth estimation model.
        intensity_pct: Intensity 0-100%.
        num_steps: Denoising steps.
        seed_sweep: Whether to try multiple seeds.
        seeds: List of seeds to try (default: [42, 123, 456]).
        validate: Whether to run input validation.

    Returns:
        PipelineResult or None if validation fails.
    """
    if seeds is None:
        seeds = SEEDS if seed_sweep else [42]

    # Input validation
    if validate:
        val = validate_input(input_bgr)
        if not val.valid:
            log.warning("Input validation failed: %s", val.message)
            return None

    # Extract landmarks
    landmarks = extract_landmarks(input_bgr)
    if landmarks is None:
        log.warning("No face detected")
        return None

    input_pil = Image.fromarray(cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB))

    # Generate procedure-specific mask
    if procedure == "blepharoplasty" and landmarks is not None:
        mask = generate_adaptive_bleph_mask(landmarks, _active_bleph_mask_config(), intensity_pct)
    elif procedure == "rhytidectomy" and landmarks is not None:
        mask = generate_adaptive_rhytid_mask(
            landmarks, _active_rhyt_mask_config(), include_neck=False,
        )
    else:
        mask = generate_mask(landmarks, procedure, MaskConfig(dilation_px=25, feather_sigma=15))

    # TPS pre-warp
    try:
        warped = apply_surgical_tps_warp(input_bgr, landmarks, procedure)
        warped_pil = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    except Exception as e:
        log.warning("TPS warp failed: %s", e)
        warped_pil = input_pil

    # Depth estimation and modification
    depth_original = depth_estimator.estimate(input_pil)
    depth_modified = modify_depth(
        depth_original, landmarks, mask, procedure, intensity_pct=intensity_pct,
    )

    # Build prompt
    prompt = build_adaptive_prompt(procedure, landmarks)

    # Inpainting strength
    if procedure == "blepharoplasty":
        strength = 0.3 + 0.25 * (intensity_pct / 100.0)
    elif procedure == "rhytidectomy":
        strength = 0.65
    else:
        strength = 0.65 + 0.20 * (intensity_pct / 100.0)

    # Seed sweep
    best_result = None
    best_score = -1.0
    best_seed = seeds[0]

    for seed in seeds:
        try:
            result_pil = run_single_seed(
                pipe, has_controlnet, warped_pil, mask,
                depth_modified, prompt, procedure, seed,
                num_steps=num_steps,
                inpainting_strength=strength,
            )
            result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)

            score = compute_arcface_score(input_bgr, result_bgr)
            log.info("Seed %d: ArcFace=%.3f", seed, score)

            if not np.isnan(score) and score > best_score:
                best_score = score
                best_result = result_bgr
                best_seed = seed
            elif best_result is None:
                best_result = result_bgr
                best_seed = seed

        except Exception as e:
            log.warning("Seed %d failed: %s", seed, e)

    if best_result is None:
        log.error("All seeds failed")
        return None

    log.info(
        "Pipeline complete: procedure=%s seed=%d ArcFace=%.3f",
        procedure, best_seed, best_score,
    )

    return PipelineResult(
        prediction=best_result,
        mask=mask,
        depth_original=depth_original,
        depth_modified=depth_modified,
        arcface_score=best_score,
        seed_used=best_seed,
        procedure=procedure,
        landmarks=landmarks,
    )
