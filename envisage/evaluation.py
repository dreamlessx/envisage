"""Decomposed evaluation metrics for surgical outcome prediction.

Novel contribution: Region-decomposed ArcFace similarity.
Instead of computing identity similarity on the full face only,
we decompose into three regions:
  1. Full face (standard ArcFace)
  2. Surgical region only (cropped to mask bounding box)
  3. Non-surgical region (everything outside the mask)

This decomposition reveals whether the model:
  - Preserves identity in untouched regions (should be ~1.0)
  - Produces realistic changes in the surgical region
  - Introduces artifacts that hurt global identity

Additional metrics:
  - DISTS (Deep Image Structure and Texture Similarity)
  - KID (Kernel Inception Distance) for batch evaluation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)


@dataclass
class DecomposedScore:
    """Region-decomposed evaluation scores."""

    full_face: float
    surgical_region: float
    non_surgical_region: float
    name: str = ""


@dataclass
class SurgicalScore:
    """Mask-decomposed eval: outside vs input, inside vs GT.

    Outside the mask: prediction must match the INPUT (identity preservation).
    Inside the mask: prediction must match the GROUND TRUTH (surgical accuracy).
    """

    outside_ssim: float    # prediction vs input, outside mask (target: ~1.0)
    outside_psnr: float    # prediction vs input, outside mask (target: 45+ dB)
    inside_lpips: float    # prediction vs GT, inside mask (lower = better)
    inside_ssim: float     # prediction vs GT, inside mask (higher = better)
    full_arcface: float    # ArcFace(input, prediction); identity preservation
    gt_arcface: float      # ArcFace(GT, prediction); how close to real outcome
    baseline_arcface: float  # ArcFace(GT, input); how much surgery changed identity
    name: str = ""

    @property
    def aggregate(self) -> float:
        """Single aggregate score out of 100.

        Components (total = 100):
          - Outside preservation (30 pts): SSIM >= 0.99 gets full marks.
            Penalized steeply below 0.98. Binary-ish: either you preserve or you don't.
          - Inside surgical accuracy (40 pts): (1 - inside_lpips) scaled.
            LPIPS 0.0 = perfect match to GT = 40 pts. LPIPS 0.5 = 20 pts.
          - Identity vs baseline (30 pts): how close is our ArcFace to the
            baseline (GT vs input). If we match or exceed baseline, full marks.
            Penalized proportionally below.

        NaN values score 0 for their component.
        """
        # Outside preservation (30 pts)
        ssim = self.outside_ssim if not np.isnan(self.outside_ssim) else 0.0
        if ssim >= 0.99:
            outside_pts = 30.0
        elif ssim >= 0.98:
            outside_pts = 25.0
        elif ssim >= 0.95:
            outside_pts = 15.0
        else:
            outside_pts = max(0.0, ssim * 30.0)

        # Inside surgical accuracy (40 pts)
        lpips = self.inside_lpips if not np.isnan(self.inside_lpips) else 1.0
        inside_pts = max(0.0, (1.0 - lpips) * 40.0)

        # Identity relative to baseline (30 pts)
        arc = self.full_arcface if not np.isnan(self.full_arcface) else 0.0
        base = self.baseline_arcface if not np.isnan(self.baseline_arcface) else 0.7
        if base < 0.1:
            base = 0.7  # fallback
        ratio = min(arc / base, 1.5)  # cap at 1.5x baseline
        identity_pts = min(30.0, ratio * 20.0)  # 1.0 ratio = 20pts, 1.5 = 30pts

        return round(outside_pts + inside_pts + identity_pts, 1)


# ---------------------------------------------------------------------------
# ArcFace helpers
# ---------------------------------------------------------------------------

_arcface_app = None


def _get_arcface():
    """Lazy-load InsightFace app."""
    global _arcface_app
    if _arcface_app is not None:
        return _arcface_app

    from insightface.app import FaceAnalysis

    # Force InsightFace onto CPU: when FLUX diffusion pipes are resident
    # in VRAM, ONNX Runtime fails to allocate even 30MB on CUDA and the
    # whole ArcFace path crashes. CPU inference runs in ~200ms per image
    # which is fine for our scoring loop.
    providers = ["CPUExecutionProvider"]
    app = FaceAnalysis(
        name="buffalo_l",
        root=str(Path.home() / ".insightface"),
        providers=providers,
    )
    # 320x320 matches pipeline.py's compute_arcface_score and works for
    # the 256-512px face crops Envisage produces. InsightFace's default
    # 640x640 missed 48/57 cases in the v2_ensemble eval on the same
    # images that detect cleanly at 320x320.
    app.prepare(ctx_id=-1, det_size=(320, 320))
    _arcface_app = app
    return app


def _get_embedding(image: np.ndarray) -> np.ndarray | None:
    """Extract ArcFace embedding from a BGR image.

    Retries on a mild upscale if the first pass finds no face; this
    catches edge cases where the subject is cropped tight or the
    resolution is below the detector's comfort range.
    """
    app = _get_arcface()
    faces = app.get(image)
    if not faces and image is not None and image.size > 0:
        # Upscale 2x and retry: detection models prefer faces > ~80px
        import cv2 as _cv2
        h, w = image.shape[:2]
        upscaled = _cv2.resize(image, (w * 2, h * 2), interpolation=_cv2.INTER_CUBIC)
        faces = app.get(upscaled)
    if not faces:
        return None
    return faces[0].embedding


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two embeddings."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------
# Region extraction
# ---------------------------------------------------------------------------

def _mask_bbox(mask: np.ndarray, pad: int = 20) -> tuple[int, int, int, int]:
    """Get bounding box of nonzero region in mask."""
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    ys, xs = np.where(mask > 0.5 if mask.dtype == np.float32 else mask > 127)
    if len(ys) == 0:
        h, w = mask.shape[:2]
        return 0, 0, w, h
    y1, y2 = max(ys.min() - pad, 0), min(ys.max() + pad, mask.shape[0])
    x1, x2 = max(xs.min() - pad, 0), min(xs.max() + pad, mask.shape[1])
    # Ensure minimum crop size for ArcFace detection
    min_size = 112
    if (y2 - y1) < min_size:
        cy = (y1 + y2) // 2
        y1 = max(cy - min_size // 2, 0)
        y2 = min(y1 + min_size, mask.shape[0])
    if (x2 - x1) < min_size:
        cx = (x1 + x2) // 2
        x1 = max(cx - min_size // 2, 0)
        x2 = min(x1 + min_size, mask.shape[1])
    return x1, y1, x2, y2


def _crop_region(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop image to bounding box."""
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2].copy()


def _mask_out_region(
    image: np.ndarray,
    mask: np.ndarray,
    fill_value: int = 128,
) -> np.ndarray:
    """Zero out pixels inside the mask (keep non-surgical region)."""
    result = image.copy()
    if mask.ndim == 2:
        mask_bool = mask > (0.5 if mask.dtype == np.float32 else 127)
        mask_3ch = np.stack([mask_bool] * 3, axis=-1)
    else:
        mask_3ch = mask > (0.5 if mask.dtype == np.float32 else 127)
    result[mask_3ch] = fill_value
    return result


# ---------------------------------------------------------------------------
# Decomposed ArcFace
# ---------------------------------------------------------------------------

def decomposed_arcface(
    input_image: np.ndarray,
    output_image: np.ndarray,
    mask: np.ndarray,
    name: str = "",
) -> DecomposedScore:
    """Compute region-decomposed ArcFace similarity.

    Args:
        input_image: BGR input face image.
        output_image: BGR output (predicted) face image.
        mask: (H, W) mask where surgical region > 0.
              Float32 [0,1] or uint8 [0,255].
        name: Optional label for this sample.

    Returns:
        DecomposedScore with full_face, surgical_region, non_surgical_region.
    """
    # Ensure same size
    h, w = input_image.shape[:2]
    if output_image.shape[:2] != (h, w):
        output_image = cv2.resize(output_image, (w, h))
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h))

    # 1. Full face ArcFace
    emb_in_full = _get_embedding(input_image)
    emb_out_full = _get_embedding(output_image)
    full_score = (
        _cosine_sim(emb_in_full, emb_out_full)
        if emb_in_full is not None and emb_out_full is not None
        else float("nan")
    )

    # 2. Surgical region (crop to mask bbox, pad with context)
    bbox = _mask_bbox(mask, pad=40)
    in_crop = _crop_region(input_image, bbox)
    out_crop = _crop_region(output_image, bbox)
    # Resize crop to at least 256x256 for ArcFace
    min_dim = 256
    ch, cw = in_crop.shape[:2]
    if ch < min_dim or cw < min_dim:
        scale = max(min_dim / ch, min_dim / cw)
        new_size = (int(cw * scale), int(ch * scale))
        in_crop = cv2.resize(in_crop, new_size)
        out_crop = cv2.resize(out_crop, new_size)

    emb_in_surg = _get_embedding(in_crop)
    emb_out_surg = _get_embedding(out_crop)
    surg_score = (
        _cosine_sim(emb_in_surg, emb_out_surg)
        if emb_in_surg is not None and emb_out_surg is not None
        else float("nan")
    )

    # 3. Non-surgical region (mask out surgical area)
    in_nonsurg = _mask_out_region(input_image, mask)
    out_nonsurg = _mask_out_region(output_image, mask)
    emb_in_ns = _get_embedding(in_nonsurg)
    emb_out_ns = _get_embedding(out_nonsurg)
    nonsurg_score = (
        _cosine_sim(emb_in_ns, emb_out_ns)
        if emb_in_ns is not None and emb_out_ns is not None
        else float("nan")
    )

    result = DecomposedScore(
        full_face=full_score,
        surgical_region=surg_score,
        non_surgical_region=nonsurg_score,
        name=name,
    )
    log.info(
        "DecomposedArcFace[%s]: full=%.4f surg=%.4f non-surg=%.4f",
        name, full_score, surg_score, nonsurg_score,
    )
    return result


# ---------------------------------------------------------------------------
# Surgical evaluation (mask-decomposed: outside vs input, inside vs GT)
# ---------------------------------------------------------------------------

_lpips_model = None


def _get_lpips():
    """Lazy-load LPIPS (AlexNet)."""
    global _lpips_model
    if _lpips_model is not None:
        return _lpips_model

    import lpips

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = lpips.LPIPS(net="alex").to(device)
    model.eval()
    _lpips_model = model
    return model


def _compute_lpips(img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> float:
    """LPIPS between two BGR uint8 images. Lower = more similar."""
    try:
        model = _get_lpips()
        device = next(model.parameters()).device
        t1 = _to_tensor(img1_bgr).to(device) * 2.0 - 1.0  # [0,1] -> [-1,1]
        t2 = _to_tensor(img2_bgr).to(device) * 2.0 - 1.0
        with torch.no_grad():
            score = model(t1, t2)
        return float(score.item())
    except Exception as e:
        log.warning("LPIPS failed: %s", e)
        return float("nan")


def _masked_psnr(
    img1: np.ndarray,
    img2: np.ndarray,
    mask_bool: np.ndarray,
) -> float:
    """PSNR computed only on pixels where mask_bool is True."""
    px1 = img1[mask_bool].astype(np.float64)
    px2 = img2[mask_bool].astype(np.float64)
    if len(px1) == 0:
        return float("nan")
    mse = np.mean((px1 - px2) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def _masked_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    mask_bool: np.ndarray,
) -> float:
    """SSIM averaged only over pixels where mask_bool is True.

    Computes the full SSIM map then averages over the mask.
    """
    from skimage.metrics import structural_similarity

    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2

    if g1.shape != g2.shape:
        g2 = cv2.resize(g2, (g1.shape[1], g1.shape[0]))

    # Resize mask if needed
    m = mask_bool
    if m.shape != g1.shape:
        m = cv2.resize(m.astype(np.uint8), (g1.shape[1], g1.shape[0])) > 0

    _, ssim_map = structural_similarity(g1, g2, full=True, data_range=255)
    masked_vals = ssim_map[m]
    if len(masked_vals) == 0:
        return float("nan")
    return float(np.mean(masked_vals))


def surgical_eval(
    input_bgr: np.ndarray,
    prediction_bgr: np.ndarray,
    gt_bgr: np.ndarray,
    mask: np.ndarray,
    name: str = "",
) -> SurgicalScore:
    """Mask-decomposed evaluation against both input and ground truth.

    Outside the mask: compare prediction vs INPUT (should be identical).
    Inside the mask: compare prediction vs GROUND TRUTH (surgical accuracy).

    Args:
        input_bgr: Original pre-op image (BGR uint8).
        prediction_bgr: Model prediction (BGR uint8).
        gt_bgr: Ground truth post-op image (BGR uint8).
        mask: (H, W) surgical mask. Float32 [0,1] or uint8 [0,255].
        name: Sample identifier.

    Returns:
        SurgicalScore with outside and inside metrics.
    """
    h, w = input_bgr.shape[:2]
    pred = prediction_bgr
    gt = gt_bgr

    # Ensure matching sizes
    if pred.shape[:2] != (h, w):
        pred = cv2.resize(pred, (w, h))
    if gt.shape[:2] != (h, w):
        gt = cv2.resize(gt, (w, h))
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h))

    # Binarize mask
    if mask.dtype == np.float32 or mask.dtype == np.float64:
        mask_bool = mask > 0.5
    else:
        mask_bool = mask > 127

    if mask_bool.ndim == 3:
        mask_bool = mask_bool[:, :, 0]

    outside_bool = ~mask_bool
    inside_bool = mask_bool

    # Expand to 3-channel for pixel indexing
    outside_3ch = np.stack([outside_bool] * 3, axis=-1)
    inside_3ch = np.stack([inside_bool] * 3, axis=-1)

    # --- Outside: prediction vs input ---
    outside_ssim = _masked_ssim(pred, input_bgr, outside_bool)
    outside_psnr = _masked_psnr(pred, input_bgr, outside_3ch)

    # --- Inside: prediction vs GT ---
    # Crop to mask bbox, then fill non-surgical pixels with neutral gray
    # so LPIPS only scores the surgical region (Codex fix)
    bbox = _mask_bbox(mask, pad=10)
    pred_crop = _crop_region(pred, bbox)
    gt_crop = _crop_region(gt, bbox)
    mask_crop = _crop_region(mask if mask.ndim == 2 else mask[:, :, 0], bbox)

    # Binarize crop mask
    if mask_crop.dtype == np.float32 or mask_crop.dtype == np.float64:
        crop_mask_bool = mask_crop > 0.5
    else:
        crop_mask_bool = mask_crop > 127

    # Fill non-surgical pixels with gray (128) in both crops
    neutral = 128
    crop_outside = ~crop_mask_bool
    if crop_outside.ndim == 2:
        crop_outside_3ch = np.stack([crop_outside] * 3, axis=-1)
    else:
        crop_outside_3ch = crop_outside
    pred_crop[crop_outside_3ch] = neutral
    gt_crop[crop_outside_3ch] = neutral

    # Resize crops to at least 64x64 for LPIPS
    ch, cw = pred_crop.shape[:2]
    if ch < 64 or cw < 64:
        scale = max(64 / ch, 64 / cw)
        new_size = (max(int(cw * scale), 64), max(int(ch * scale), 64))
        pred_crop = cv2.resize(pred_crop, new_size)
        gt_crop = cv2.resize(gt_crop, new_size)

    inside_lpips = _compute_lpips(pred_crop, gt_crop)
    inside_ssim = _masked_ssim(pred, gt, inside_bool)

    # --- ArcFace ---
    emb_input = _get_embedding(input_bgr)
    emb_pred = _get_embedding(pred)
    emb_gt = _get_embedding(gt)

    full_arcface = (
        _cosine_sim(emb_input, emb_pred)
        if emb_input is not None and emb_pred is not None
        else float("nan")
    )
    gt_arcface = (
        _cosine_sim(emb_gt, emb_pred)
        if emb_gt is not None and emb_pred is not None
        else float("nan")
    )
    baseline_arcface = (
        _cosine_sim(emb_gt, emb_input)
        if emb_gt is not None and emb_input is not None
        else float("nan")
    )

    result = SurgicalScore(
        outside_ssim=outside_ssim,
        outside_psnr=outside_psnr,
        inside_lpips=inside_lpips,
        inside_ssim=inside_ssim,
        full_arcface=full_arcface,
        gt_arcface=gt_arcface,
        baseline_arcface=baseline_arcface,
        name=name,
    )
    log.info(
        "SurgicalScore[%s]: out_ssim=%.4f out_psnr=%.1f in_lpips=%.4f "
        "in_ssim=%.4f arcface=%.4f gt_arc=%.4f base_arc=%.4f",
        name, outside_ssim, outside_psnr, inside_lpips,
        inside_ssim, full_arcface, gt_arcface, baseline_arcface,
    )
    return result


def format_surgical_results(scores: list[SurgicalScore]) -> str:
    """Format SurgicalScore list as a table."""
    lines = [
        f"{'Name':<25} {'Out SSIM':<10} {'Out PSNR':<10} "
        f"{'In LPIPS':<10} {'In SSIM':<10} {'ArcFace':<10} "
        f"{'GT Arc':<10} {'Base Arc':<10} {'Score':<7}",
        "-" * 102,
    ]
    for s in scores:
        lines.append(
            f"{s.name:<25} {s.outside_ssim:<10.4f} {s.outside_psnr:<10.1f} "
            f"{s.inside_lpips:<10.4f} {s.inside_ssim:<10.4f} "
            f"{s.full_arcface:<10.4f} {s.gt_arcface:<10.4f} "
            f"{s.baseline_arcface:<10.4f} {s.aggregate:<7.1f}"
        )

    def _nanmean(vals):
        v = [x for x in vals if not np.isnan(x)]
        return np.mean(v) if v else float("nan")

    lines.append("-" * 102)
    lines.append(
        f"{'MEAN':<25} "
        f"{_nanmean([s.outside_ssim for s in scores]):<10.4f} "
        f"{_nanmean([s.outside_psnr for s in scores]):<10.1f} "
        f"{_nanmean([s.inside_lpips for s in scores]):<10.4f} "
        f"{_nanmean([s.inside_ssim for s in scores]):<10.4f} "
        f"{_nanmean([s.full_arcface for s in scores]):<10.4f} "
        f"{_nanmean([s.gt_arcface for s in scores]):<10.4f} "
        f"{_nanmean([s.baseline_arcface for s in scores]):<10.4f} "
        f"{np.mean([s.aggregate for s in scores]):<7.1f}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DISTS metric
# ---------------------------------------------------------------------------

def compute_dists(
    input_image: torch.Tensor | np.ndarray,
    output_image: torch.Tensor | np.ndarray,
) -> float:
    """Compute DISTS (Deep Image Structure and Texture Similarity).

    Args:
        input_image: (3, H, W) or (H, W, 3) float32 in [0, 1] or uint8.
        output_image: Same format.

    Returns:
        DISTS score (lower = more similar).
    """
    try:
        from piq import DISTS as DISTSMetric

        x = _to_tensor(input_image)
        y = _to_tensor(output_image)
        dists = DISTSMetric()
        score = dists(x, y)
        return float(score.item())
    except ImportError:
        log.warning("piq not installed, DISTS unavailable")
        return float("nan")


# ---------------------------------------------------------------------------
# KID metric (batch)
# ---------------------------------------------------------------------------

def compute_kid(
    real_images: list[np.ndarray],
    generated_images: list[np.ndarray],
) -> float:
    """Compute KID (Kernel Inception Distance) between image sets.

    Args:
        real_images: List of BGR uint8 images.
        generated_images: List of BGR uint8 images.

    Returns:
        KID score (lower = better).
    """
    try:
        from piq import KID

        real_tensors = torch.stack([_to_tensor(img).squeeze(0) for img in real_images])
        gen_tensors = torch.stack([_to_tensor(img).squeeze(0) for img in generated_images])

        # KID needs at least some samples
        if len(real_tensors) < 2 or len(gen_tensors) < 2:
            log.warning("KID needs at least 2 samples per set")
            return float("nan")

        kid = KID()
        kid.update(real_tensors, real=True)
        kid.update(gen_tensors, real=False)
        score = kid.compute()
        return float(score[0].item())  # (mean, std)
    except ImportError:
        log.warning("piq not installed, KID unavailable")
        return float("nan")
    except Exception as e:
        log.warning("KID computation failed: %s", e)
        return float("nan")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_tensor(image: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert image to (1, 3, H, W) float32 tensor in [0, 1]."""
    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            image = image.unsqueeze(0)
        return image.float()

    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    if image.ndim == 3 and image.shape[2] == 3:
        # (H, W, 3) BGR -> (1, 3, H, W) RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2, 0, 1)

    return torch.from_numpy(image).unsqueeze(0).float()


def format_results(scores: list[DecomposedScore]) -> str:
    """Format decomposed scores as a table."""
    lines = [
        f"{'Name':<20} {'Full':<10} {'Surgical':<10} {'Non-Surg':<10}",
        "-" * 50,
    ]
    for s in scores:
        lines.append(
            f"{s.name:<20} {s.full_face:<10.4f} {s.surgical_region:<10.4f} "
            f"{s.non_surgical_region:<10.4f}"
        )

    # Mean
    full_mean = np.nanmean([s.full_face for s in scores])
    surg_mean = np.nanmean([s.surgical_region for s in scores])
    ns_mean = np.nanmean([s.non_surgical_region for s in scores])
    lines.append("-" * 50)
    lines.append(f"{'MEAN':<20} {full_mean:<10.4f} {surg_mean:<10.4f} {ns_mean:<10.4f}")

    return "\n".join(lines)
