"""Composite quality scorer for candidate outputs.

The decision model. Every candidate goes through seven hard gates
(identity, outside-SSIM, landmark drift, dark-hole, color-shift,
bleph-crease, and procedure-fidelity). Disqualified candidates
cannot be selected. Surviving candidates are ranked by a weighted
composite score; the top-scored candidate is shipped. If ALL
diffusion candidates (M1-M4) fail their gates, the deterministic TPS
output (M5) is shipped unconditionally -- the zero-hallucination
guarantee.

Hard gates (all must pass):
  identity:  ArcFace(input, cand)           >= 0.65
  outside:   SSIM(input, cand)[outside]     >= 0.95
  landmark:  median drift inside mask       <= 15.0 px
  dark-hole: V-channel < 20                 <= 0.5% area
  color:     mean hue shift                 <= 15.0 deg
  crease:    |Sobel-y| (bleph only)         >= 12.0
  fidelity:  measurement deltas             within severity bands

Composite score (weights sum to 1.0):
  0.25 identity            0.15 cross-method agreement
  0.15 outside             0.10 TPS agreement
  0.15 landmark            0.05 aesthetic
  0.15 color consistency
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thresholds + weights (tunable; sweep in Plan Phase 3)
# ---------------------------------------------------------------------------

IDENTITY_MIN_ARCFACE: float = 0.65
# 0.95 instead of 0.98 because `apply_hard_mask_composite` uses a
# feathered mask (soft boundary), so SSIM outside the mask includes a
# few blended boundary pixels. 0.95 is still tight enough to catch
# any real model bleed beyond the mask.
OUTSIDE_MIN_SSIM: float = 0.95
LANDMARK_MAX_DRIFT_PX: float = 15.0

# Anti-hallucination gates inside the mask region.
# Rhino failure mode: nostril-black holes, color drift on skin patches.
# Bleph failure mode: flat upper lid with no visible crease edge.
DARK_HOLE_MAX_PIXEL_FRACTION: float = 0.005   # >0.5% of mask going below V-threshold disqualifies
DARK_HOLE_BRIGHTNESS_THRESHOLD: int = 20       # V channel [0,255]; below this counts as dark
# Tightened 2% -> 0.5% per Mudit iter 2026-04-18 rhino audit: Nose_27 shipped
# with a visible dark nostril hallucination that slid under the previous
# threshold. Tighter gate forces the TPS fallback on artifact cases.
COLOR_MAX_HUE_SHIFT: float = 15.0              # mean hue shift (degrees, 0-180) inside mask
BLEPH_CREASE_MIN_EDGE_STRENGTH: float = 12.0   # Sobel |gy| mean along expected crease line

# Procedure-fidelity gate: wide tolerance bands to avoid disqualifying
# reasonable surgical candidates. Multipliers apply to each preset's
# `delta_threshold` (baseline atlas-magnitude for MODERATE severity).
# Empirically, real diffusion outputs (M2 ICEdit, M3 Kontext, M4 Fill)
# produce deltas 0.05x-0.3x of the atlas magnitude on rhino, so the
# MILD minimum at 0.1x admits genuinely subtle real edits.
FIDELITY_SEVERITY_MIN_MULT: dict[int, float] = {1: 0.1, 2: 0.4, 3: 0.8}  # MILD, MODERATE, SEVERE
FIDELITY_SEVERITY_MAX_MULT: dict[int, float] = {1: 3.0, 2: 4.5, 3: 8.0}
FIDELITY_INACTIVE_TOL_MULT: float = 1.2    # inactive presets may drift within this
FIDELITY_TEXTURE_TOL_MULT: float = 1.5     # texture-only (expected_sign=0) active presets

WEIGHTS: dict[str, float] = {
    "identity": 0.25,
    "outside": 0.15,
    "landmark": 0.15,
    "color_consistency": 0.15,
    "cross_agreement": 0.15,
    "tps_agreement": 0.10,
    "aesthetic": 0.05,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

TPS_METHOD: str = "M5_tps"


@dataclass(frozen=True)
class Candidate:
    """A single generated candidate output."""

    image_bgr: np.ndarray
    method: str          # "M1_flux_zs" | "M2_icedit" | "M3_kontext" | "M4_fill_lora" | "M5_tps"
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_fallback(self) -> bool:
        return self.method == TPS_METHOD


@dataclass
class CandidateScore:
    """Score decomposition for a single candidate."""

    method: str
    seed: int

    # Raw gate measurements
    identity_arcface: float
    outside_ssim: float
    landmark_drift_px: float
    dark_hole_fraction: float     # fraction of mask pixels below brightness threshold
    color_hue_shift: float        # mean hue delta (deg) inside mask vs input skin
    bleph_crease_edge: float      # Sobel |gy| along expected crease (bleph only; else nan)

    # Normalized [0, 1] component scores
    identity_score: float
    outside_score: float
    landmark_score: float
    color_score: float
    cross_agreement: float
    tps_agreement: float
    aesthetic: float

    # Procedure-aware flag; set by score_candidate when procedure == "blepharoplasty"
    check_crease: bool = False

    # Procedure-fidelity gate: did the candidate show the measurement deltas
    # the active presets demand? Missing (no Analysis supplied) means the
    # gate is skipped and `fidelity_passes` is True by default -- we never
    # disqualify on missing fidelity data.
    fidelity_checked: bool = False
    fidelity_passes: bool = True
    fidelity_failures: tuple[str, ...] = ()    # e.g. ("dorsal_hump_reduction: delta=+0.1 outside [-3.0,-0.3]",)

    @property
    def identity_passes(self) -> bool:
        return not np.isnan(self.identity_arcface) and self.identity_arcface >= IDENTITY_MIN_ARCFACE

    @property
    def outside_passes(self) -> bool:
        return not np.isnan(self.outside_ssim) and self.outside_ssim >= OUTSIDE_MIN_SSIM

    @property
    def landmark_passes(self) -> bool:
        return (not np.isinf(self.landmark_drift_px)
                and self.landmark_drift_px <= LANDMARK_MAX_DRIFT_PX)

    @property
    def dark_hole_passes(self) -> bool:
        return (not np.isnan(self.dark_hole_fraction)
                and self.dark_hole_fraction <= DARK_HOLE_MAX_PIXEL_FRACTION)

    @property
    def color_passes(self) -> bool:
        return (not np.isnan(self.color_hue_shift)
                and self.color_hue_shift <= COLOR_MAX_HUE_SHIFT)

    @property
    def crease_passes(self) -> bool:
        """Bleph-only crease presence. For other procedures, always passes."""
        if not self.check_crease:
            return True
        if np.isnan(self.bleph_crease_edge):
            return False
        return self.bleph_crease_edge >= BLEPH_CREASE_MIN_EDGE_STRENGTH

    @property
    def disqualified(self) -> bool:
        return not (self.identity_passes
                    and self.outside_passes
                    and self.landmark_passes
                    and self.dark_hole_passes
                    and self.color_passes
                    and self.crease_passes
                    and self.fidelity_passes)

    @property
    def disqualify_reasons(self) -> list[str]:
        r: list[str] = []
        if not self.identity_passes:
            r.append(f"identity={self.identity_arcface:.3f}<{IDENTITY_MIN_ARCFACE}")
        if not self.outside_passes:
            r.append(f"outside_ssim={self.outside_ssim:.3f}<{OUTSIDE_MIN_SSIM}")
        if not self.landmark_passes:
            r.append(f"landmark_drift={self.landmark_drift_px:.1f}>{LANDMARK_MAX_DRIFT_PX}")
        if not self.dark_hole_passes:
            r.append(f"dark_hole_frac={self.dark_hole_fraction:.3f}>{DARK_HOLE_MAX_PIXEL_FRACTION}")
        if not self.color_passes:
            r.append(f"hue_shift={self.color_hue_shift:.1f}>{COLOR_MAX_HUE_SHIFT}")
        if self.check_crease and not self.crease_passes:
            r.append(f"crease_edge={self.bleph_crease_edge:.1f}<{BLEPH_CREASE_MIN_EDGE_STRENGTH}")
        if self.fidelity_checked and not self.fidelity_passes:
            r.extend(f"fidelity: {f}" for f in self.fidelity_failures)
        return r

    @property
    def composite(self) -> float:
        """Weighted composite. -inf if disqualified."""
        if self.disqualified:
            return float("-inf")
        return (
            WEIGHTS["identity"] * self.identity_score
            + WEIGHTS["outside"] * self.outside_score
            + WEIGHTS["landmark"] * self.landmark_score
            + WEIGHTS["color_consistency"] * self.color_score
            + WEIGHTS["cross_agreement"] * self.cross_agreement
            + WEIGHTS["tps_agreement"] * self.tps_agreement
            + WEIGHTS["aesthetic"] * self.aesthetic
        )


# ---------------------------------------------------------------------------
# Internal helpers: reuse evaluation.py where possible
# ---------------------------------------------------------------------------

def _arcface_similarity(img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> float:
    from .evaluation import _cosine_sim, _get_embedding
    a = _get_embedding(img1_bgr)
    b = _get_embedding(img2_bgr)
    if a is None or b is None:
        return float("nan")
    return _cosine_sim(a, b)


def _binarize_mask(mask: np.ndarray) -> np.ndarray:
    """Binarize 2D/3D mask to a 2D bool array."""
    if mask.dtype in (np.float32, np.float64):
        mb = mask > 0.5
    else:
        mb = mask > 127
    if mb.ndim == 3:
        mb = mb[:, :, 0]
    return mb


def _outside_mask_ssim(pred_bgr: np.ndarray, input_bgr: np.ndarray, mask: np.ndarray) -> float:
    from .evaluation import _masked_ssim
    try:
        return _masked_ssim(pred_bgr, input_bgr, ~_binarize_mask(mask))
    except Exception as e:
        log.warning("outside_ssim failed: %s", e)
        return float("nan")


def _landmark_drift_inside_mask(
    pred_bgr: np.ndarray,
    input_bgr: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Median pixel drift of landmarks inside the mask, pred vs input.

    Catches hallucinations where the output looks like a face but the
    surgical region has migrated relative to where the mask constrained it.
    """
    from .landmarks import extract_landmarks

    lm_in = extract_landmarks(input_bgr)
    lm_out = extract_landmarks(pred_bgr)
    if lm_in is None or lm_out is None:
        log.debug("landmark drift: extraction returned None")
        return float("inf")

    n = min(len(lm_in.points), len(lm_out.points))
    if n == 0:
        return float("inf")

    pts_in = lm_in.points[:n]
    pts_out = lm_out.points[:n]

    mb = _binarize_mask(mask)
    h, w = mb.shape[:2]

    inside_idx: list[int] = []
    for i in range(n):
        x = int(np.clip(pts_in[i, 0], 0, w - 1))
        y = int(np.clip(pts_in[i, 1], 0, h - 1))
        if mb[y, x]:
            inside_idx.append(i)

    if not inside_idx:
        # No landmarks inside mask; cannot measure drift, treat as pass
        return 0.0

    drifts = np.linalg.norm(pts_out[inside_idx] - pts_in[inside_idx], axis=1)
    return float(np.median(drifts))


def _gray_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Grayscale full-image SSIM, tolerant to shape mismatch."""
    from skimage.metrics import structural_similarity
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.ndim == 3 else a
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) if b.ndim == 3 else b
    if gb.shape != ga.shape:
        gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]))
    return float(structural_similarity(ga, gb, data_range=255))


# DINOv2 lazy-loaded singleton. Heavy (~86M params) so only instantiated
# once per process. Fallback to SSIM proxy if transformers / torch are not
# available at runtime -- keeps the scorer functional on CPU-only machines.
_dino_model = None
_dino_processor = None
_dino_load_failed = False


def _load_dino():
    """Lazy-load DINOv2 ViT-B/14 (facebook/dinov2-base).

    Returns (model, processor, device) on success, None on failure.
    Failures are sticky: once _dino_load_failed is set, subsequent calls
    skip the load attempt and callers fall back to the SSIM proxy.
    """
    global _dino_model, _dino_processor, _dino_load_failed

    if _dino_load_failed:
        return None
    if _dino_model is not None and _dino_processor is not None:
        return _dino_model, _dino_processor, next(_dino_model.parameters()).device

    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
        model.eval()
        _dino_model = model
        _dino_processor = processor
        log.info("DINOv2 ViT-B/14 loaded on %s", device)
        return model, processor, device
    except Exception as e:
        log.warning("DINOv2 load failed (%s); cross_method_agreement will use SSIM proxy", e)
        _dino_load_failed = True
        return None


def _dino_embedding(image_bgr: np.ndarray) -> np.ndarray | None:
    """Extract the CLS-token embedding from DINOv2 ViT-B/14. Returns None on failure."""
    import torch

    loaded = _load_dino()
    if loaded is None:
        return None
    model, processor, device = loaded

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) if image_bgr.ndim == 3 else image_bgr
    try:
        inputs = processor(images=rgb, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        cls = outputs.last_hidden_state[:, 0, :]          # (1, D)
        vec = cls.cpu().numpy()[0].astype(np.float32)
        n = np.linalg.norm(vec)
        return vec / n if n > 1e-12 else vec
    except Exception as e:
        log.warning("DINOv2 embedding failed: %s", e)
        return None


def _cross_method_agreement(
    cand: np.ndarray,
    peers: list[np.ndarray],
) -> float:
    """Mean DINOv2 cosine similarity of candidate vs each peer.

    Primary: DINOv2 ViT-B/14 CLS-token cosine similarity, averaged across
    peers. Falls back to grayscale-SSIM mean when DINOv2 is unavailable
    (e.g. CPU-only laptop, offline environment). The fallback is noisy
    but monotonic with agreement, so composite ordering is preserved.
    """
    if not peers:
        return 0.5

    cand_emb = _dino_embedding(cand)
    if cand_emb is None:
        return float(np.mean([_gray_ssim(cand, p) for p in peers]))

    scores: list[float] = []
    for p in peers:
        p_emb = _dino_embedding(p)
        if p_emb is None:
            scores.append(_gray_ssim(cand, p))
            continue
        scores.append(float(np.dot(cand_emb, p_emb)))

    # DINOv2 cosine is in [-1, 1]; re-normalize to [0, 1] for composite.
    mean = float(np.mean(scores)) if scores else 0.5
    return float(np.clip((mean + 1.0) / 2.0, 0.0, 1.0))


def _tps_agreement(cand: np.ndarray, tps_output: np.ndarray, mask: np.ndarray) -> float:
    """Inside-mask SSIM between candidate and TPS fallback.

    High agreement means the diffusion output is geometrically consistent
    with the deterministic TPS warp -- strong signal against hallucination.
    """
    from .evaluation import _masked_ssim
    try:
        return _masked_ssim(cand, tps_output, _binarize_mask(mask))
    except Exception as e:
        log.warning("tps_agreement failed: %s", e)
        return 0.5


def _aesthetic(image_bgr: np.ndarray) -> float:
    """CLIP/NIMA aesthetic placeholder. Returns neutral 0.5.

    Will be replaced with an actual CLIP-aesthetic head before NeurIPS.
    Kept as a hook so the composite weight slot is already wired.
    """
    return 0.5


# ---------------------------------------------------------------------------
# Anti-hallucination: dark-hole detection, color consistency, bleph crease
# ---------------------------------------------------------------------------

def _dark_hole_fraction(pred_bgr: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of mask pixels whose V-channel falls below the dark threshold.

    Flags nostril-black holes, melted features, and ambient-dark artifacts.
    Normal shadow regions in a face (nostril interior, lash-line) should be
    a small fraction of the mask; a large fraction indicates hallucination.
    """
    try:
        hsv = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        mb = _binarize_mask(mask)
        if mb.shape != v.shape:
            mb = cv2.resize(mb.astype(np.uint8), (v.shape[1], v.shape[0])) > 0
        inside = v[mb]
        if inside.size == 0:
            return 0.0
        return float(np.mean(inside < DARK_HOLE_BRIGHTNESS_THRESHOLD))
    except Exception as e:
        log.warning("dark_hole_fraction failed: %s", e)
        return float("nan")


def _skin_reference_stats(input_bgr: np.ndarray, mask: np.ndarray) -> tuple[float, float] | None:
    """Mean (H, S) of input pixels OUTSIDE the mask but within a face-area proxy.

    Used as the reference skin tone; the candidate's inside-mask mean is
    compared against this to detect color drift.
    """
    try:
        hsv = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2HSV)
        mb = _binarize_mask(mask)
        if mb.shape != hsv.shape[:2]:
            mb = cv2.resize(mb.astype(np.uint8), (hsv.shape[1], hsv.shape[0])) > 0
        # Reference = not-mask pixels with V above a threshold (drop pure black bg)
        v = hsv[:, :, 2]
        outside = (~mb) & (v > 40) & (v < 240)
        if outside.sum() < 100:
            return None
        mean_h = float(np.mean(hsv[:, :, 0][outside]))
        mean_s = float(np.mean(hsv[:, :, 1][outside]))
        return mean_h, mean_s
    except Exception:
        return None


def _color_hue_shift(
    pred_bgr: np.ndarray,
    input_bgr: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Absolute hue shift (degrees, 0-180) of mask interior vs input skin reference.

    Large values indicate color drift: the generated skin inside the mask
    is not the same hue as the rest of the face. This catches "weird
    colored patches" and hallucinated pigmentation.
    """
    ref = _skin_reference_stats(input_bgr, mask)
    if ref is None:
        return float("nan")
    ref_h, _ = ref

    try:
        hsv = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2HSV)
        mb = _binarize_mask(mask)
        if mb.shape != hsv.shape[:2]:
            mb = cv2.resize(mb.astype(np.uint8), (hsv.shape[1], hsv.shape[0])) > 0
        # Restrict to skin-like pixels (drop near-black shadows + near-white)
        v = hsv[:, :, 2]
        skin = mb & (v > 40) & (v < 240)
        if skin.sum() < 50:
            return 0.0
        pred_h = float(np.mean(hsv[:, :, 0][skin]))
        # OpenCV HSV H is in [0, 180]; use circular distance
        diff = abs(pred_h - ref_h)
        return float(min(diff, 180.0 - diff))
    except Exception as e:
        log.warning("color_hue_shift failed: %s", e)
        return float("nan")


def _bleph_crease_edge_strength(
    pred_bgr: np.ndarray,
    input_bgr: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Mean |Sobel-y| inside the mask. Higher = stronger horizontal edges.

    The supratarsal crease is a horizontal edge inside the upper-lid
    mask. If the model produces a smooth featureless lid, Sobel-y is
    weak. A normal lid with a visible crease has a clear horizontal
    gradient band.
    """
    try:
        gray = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2GRAY) if pred_bgr.ndim == 3 else pred_bgr
        mb = _binarize_mask(mask)
        if mb.shape != gray.shape:
            mb = cv2.resize(mb.astype(np.uint8), (gray.shape[1], gray.shape[0])) > 0
        if mb.sum() < 50:
            return float("nan")
        sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        return float(np.mean(sobel_y[mb]))
    except Exception as e:
        log.warning("bleph_crease_edge failed: %s", e)
        return float("nan")


# ---------------------------------------------------------------------------
# Procedure fidelity: the gate that prevents "do nothing" from winning
# ---------------------------------------------------------------------------

def _procedure_key_to_proc_dict(procedure: str) -> dict[str, Any] | None:
    """Return the {key: SubProcedure} map for a procedure, or None if unknown."""
    try:
        if procedure == "rhinoplasty":
            from .rhino_config import RHINO_PROCEDURES
            return RHINO_PROCEDURES
        if procedure == "blepharoplasty":
            from .bleph_config import BLEPH_PROCEDURES
            return BLEPH_PROCEDURES
        if procedure == "rhytidectomy":
            from .rhytid_config import RHYTID_PROCEDURES
            return RHYTID_PROCEDURES
    except Exception:
        return None
    return None


def _check_procedure_fidelity(
    candidate_bgr: np.ndarray,
    input_bgr: np.ndarray,
    procedure: str | None,
    analysis: Any | None,
) -> tuple[bool, tuple[str, ...]]:
    """Verify the candidate actually applied the active presets.

    Each active preset's `measurement_key` is measured on both input and
    candidate; the delta is compared against a severity-scaled band. If
    the delta is missing, wrong direction, or outside the magnitude
    band, the preset fails. Inactive presets must have a delta below the
    noise tolerance (catches over-editing).

    Returns (passes, failure_messages). Missing analysis or missing
    measurements never disqualifies -- the gate passes silently in that
    case so we never reject on data we couldn't compute.
    """
    if procedure is None or analysis is None:
        return True, ()

    proc_dict = _procedure_key_to_proc_dict(procedure)
    if proc_dict is None:
        return True, ()

    from .measurements import measure_all

    m_in = measure_all(input_bgr)
    m_cand = measure_all(candidate_bgr)
    if not m_in or not m_cand:
        log.info("fidelity: measurements unavailable; skipping gate")
        return True, ()

    failures: list[str] = []
    active = set(getattr(analysis, "active_keys", []) or [])
    severities = getattr(analysis, "severity", {}) or {}

    for key, proc in proc_dict.items():
        delta_thr = getattr(proc, "delta_threshold", None)
        meas_key = getattr(proc, "measurement_key", None)
        sign = getattr(proc, "expected_sign", 0)
        if delta_thr is None or meas_key is None:
            continue

        v_in = m_in.get(meas_key, float("nan"))
        v_cand = m_cand.get(meas_key, float("nan"))
        if np.isnan(v_in) or np.isnan(v_cand):
            continue  # never disqualify on missing measurement
        delta = v_cand - v_in

        if key in active:
            sev = severities.get(key, 2)  # default MODERATE if severity not set
            if sign == 0:
                # Texture-only preset: landmark delta should stay near zero.
                tol = FIDELITY_TEXTURE_TOL_MULT * delta_thr
                if abs(delta) > tol:
                    failures.append(
                        f"{key}: |delta|={abs(delta):.3f} > texture_tol={tol:.3f}"
                    )
                continue

            min_mag = FIDELITY_SEVERITY_MIN_MULT.get(sev, 0.3) * delta_thr
            max_mag = FIDELITY_SEVERITY_MAX_MULT.get(sev, 3.0) * delta_thr
            # Required band in signed terms
            if sign > 0:
                lo, hi = +min_mag, +max_mag
            else:
                lo, hi = -max_mag, -min_mag
            if not (lo <= delta <= hi):
                failures.append(
                    f"{key}: delta={delta:+.3f} outside [{lo:+.3f},{hi:+.3f}] (sev={sev})"
                )
        else:
            # Inactive preset: allow small incidental drift only.
            tol = FIDELITY_INACTIVE_TOL_MULT * delta_thr
            if abs(delta) > tol:
                failures.append(
                    f"{key}: inactive drift |delta|={abs(delta):.3f} > tol={tol:.3f}"
                )

    return (len(failures) == 0), tuple(failures)


# ---------------------------------------------------------------------------
# Hard-mask composite: guarantee byte-identical outside-mask pixels
# ---------------------------------------------------------------------------

def apply_hard_mask_composite(
    pred_bgr: np.ndarray,
    input_bgr: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Composite prediction with input using a soft mask.

    Outside the mask, output is byte-identical to input. Inside the mask,
    output is the prediction. At the feathered boundary, output is an
    alpha blend. This is the architectural identity guarantee that
    pushes outside-mask SSIM from ~0.98 to 1.000 and prevents ambient
    drift on the rest of the face.

    Args:
        pred_bgr: diffusion output (BGR uint8).
        input_bgr: original input (BGR uint8).
        mask: surgical mask, float32 [0, 1] or uint8 [0, 255]. Feathered ok.

    Returns:
        Composite BGR uint8 with same shape as input_bgr.
    """
    if pred_bgr.shape != input_bgr.shape:
        pred_bgr = cv2.resize(pred_bgr, (input_bgr.shape[1], input_bgr.shape[0]))

    m = mask
    if m.ndim == 3:
        m = m[:, :, 0]
    if m.shape != input_bgr.shape[:2]:
        m = cv2.resize(m, (input_bgr.shape[1], input_bgr.shape[0]))
    if m.dtype == np.uint8:
        alpha = m.astype(np.float32) / 255.0
    else:
        alpha = np.clip(m.astype(np.float32), 0.0, 1.0)

    alpha3 = alpha[:, :, np.newaxis]
    composite = (alpha3 * pred_bgr.astype(np.float32)
                 + (1.0 - alpha3) * input_bgr.astype(np.float32))
    return np.clip(composite, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_candidate(
    candidate: Candidate,
    input_bgr: np.ndarray,
    mask: np.ndarray,
    tps_reference_bgr: np.ndarray,
    peer_candidates: list[Candidate],
    procedure: str | None = None,
    analysis: Any | None = None,
) -> CandidateScore:
    """Score one candidate against input, mask, TPS reference, and peers.

    `procedure` unlocks procedure-specific anti-hallucination gates:
      - "blepharoplasty"  → bleph_crease_edge gate (catches flat lids)
      - other             → generic dark-hole + color-consistency gates only

    `analysis` enables the procedure-fidelity gate. When supplied (the
    RhinoAnalysis / BlephAnalysis / RhytidAnalysis for this case), the
    scorer verifies that the candidate's per-preset measurement deltas
    fall within severity-scaled bands. This is the gate that prevents
    a "return the input unchanged" candidate from passing the other
    six gates and scoring high on the composite.
    """
    pred = candidate.image_bgr

    arc = _arcface_similarity(input_bgr, pred)
    out_ssim = _outside_mask_ssim(pred, input_bgr, mask)
    drift = _landmark_drift_inside_mask(pred, input_bgr, mask)

    dark_frac = _dark_hole_fraction(pred, mask)
    hue_shift = _color_hue_shift(pred, input_bgr, mask)

    check_crease = (procedure == "blepharoplasty")
    crease = _bleph_crease_edge_strength(pred, input_bgr, mask) if check_crease else float("nan")

    # Normalize to [0, 1] for composite components
    id_s = 0.0 if np.isnan(arc) else float(np.clip(arc, 0.0, 1.0))
    out_s = 0.0 if np.isnan(out_ssim) else float(np.clip(out_ssim, 0.0, 1.0))
    lm_s = 0.0 if np.isinf(drift) else float(np.clip(1.0 - drift / (2.0 * LANDMARK_MAX_DRIFT_PX), 0.0, 1.0))

    # Color score combines hue-shift tolerance AND dark-hole absence
    hue_component = 1.0 if np.isnan(hue_shift) else float(
        np.clip(1.0 - hue_shift / (2.0 * COLOR_MAX_HUE_SHIFT), 0.0, 1.0)
    )
    hole_component = 1.0 if np.isnan(dark_frac) else float(
        np.clip(1.0 - dark_frac / (4.0 * DARK_HOLE_MAX_PIXEL_FRACTION), 0.0, 1.0)
    )
    color_s = 0.5 * hue_component + 0.5 * hole_component

    peer_imgs = [
        p.image_bgr
        for p in peer_candidates
        if not (p.method == candidate.method and p.seed == candidate.seed)
    ]
    cross = _cross_method_agreement(pred, peer_imgs)
    tps_a = _tps_agreement(pred, tps_reference_bgr, mask)
    aesth = _aesthetic(pred)

    # Procedure-fidelity gate. Skipped for the TPS fallback candidate:
    # M5 is the deterministic zero-hallucination output; it may or may
    # not hit the severity band for every preset, and we never want to
    # disqualify the fallback on a soft measurement.
    if candidate.is_fallback or analysis is None or procedure is None:
        fidelity_checked = False
        fidelity_passes = True
        fidelity_failures: tuple[str, ...] = ()
    else:
        fidelity_checked = True
        fidelity_passes, fidelity_failures = _check_procedure_fidelity(
            pred, input_bgr, procedure, analysis,
        )

    return CandidateScore(
        method=candidate.method,
        seed=candidate.seed,
        identity_arcface=arc,
        outside_ssim=out_ssim,
        landmark_drift_px=drift,
        dark_hole_fraction=dark_frac,
        color_hue_shift=hue_shift,
        bleph_crease_edge=crease,
        identity_score=id_s,
        outside_score=out_s,
        landmark_score=lm_s,
        color_score=color_s,
        cross_agreement=cross,
        tps_agreement=tps_a,
        aesthetic=aesth,
        check_crease=check_crease,
        fidelity_checked=fidelity_checked,
        fidelity_passes=fidelity_passes,
        fidelity_failures=fidelity_failures,
    )


def select_best(
    candidates: list[Candidate],
    input_bgr: np.ndarray,
    mask: np.ndarray,
    procedure: str | None = None,
    analysis: Any | None = None,
) -> tuple[Candidate, CandidateScore, list[CandidateScore]]:
    """Score all candidates, apply hard gates, return the chosen output.

    Contract:
      - `candidates` MUST include at least one candidate with method == M5_tps.
      - Returns (winner, winner_score, all_scores_in_candidate_order).
      - If no diffusion candidate (M1-M4) survives the hard gates, returns
        the TPS fallback. This is the zero-hallucination guarantee.
    """
    if not candidates:
        raise ValueError("select_best requires at least one candidate")

    tps_list = [c for c in candidates if c.is_fallback]
    if not tps_list:
        raise ValueError(
            "select_best contract violated: no TPS fallback candidate "
            f"(method == {TPS_METHOD!r}) found among {len(candidates)} candidates"
        )
    tps = tps_list[0]

    all_scores = [
        score_candidate(c, input_bgr, mask, tps.image_bgr, candidates,
                        procedure=procedure, analysis=analysis)
        for c in candidates
    ]

    diffusion_scored = [
        (c, s) for c, s in zip(candidates, all_scores)
        if not c.is_fallback and not s.disqualified
    ]

    n_diffusion = sum(1 for c in candidates if not c.is_fallback)

    if diffusion_scored:
        diffusion_scored.sort(key=lambda cs: cs[1].composite, reverse=True)
        winner_cand, winner_score = diffusion_scored[0]
        log.info(
            "select_best: shipping %s seed=%d composite=%.3f (%d/%d diffusion candidates survived)",
            winner_cand.method, winner_cand.seed, winner_score.composite,
            len(diffusion_scored), n_diffusion,
        )
        return winner_cand, winner_score, all_scores

    tps_score = next(s for c, s in zip(candidates, all_scores) if c.is_fallback)
    disq_summary = {s.method: s.disqualify_reasons for s in all_scores if s.disqualified}
    log.warning(
        "select_best: ALL %d diffusion candidates disqualified -> shipping TPS fallback. "
        "Reasons: %s",
        n_diffusion, disq_summary,
    )
    return tps, tps_score, all_scores


def format_scores(scores: list[CandidateScore]) -> str:
    """Tabular summary of candidate scores, for logs and debugging."""
    lines = [
        f"{'method':<14} {'seed':<6} {'arc':<7} {'out':<7} {'drift':<7} "
        f"{'dhole':<7} {'hue':<7} {'crease':<7} "
        f"{'cross':<7} {'tps':<7} {'comp':<7} {'disq':<5}",
        "-" * 112,
    ]
    for s in scores:
        comp = s.composite
        comp_str = "-inf" if comp == float("-inf") else f"{comp:.3f}"
        crease_str = "n/a" if not s.check_crease else f"{s.bleph_crease_edge:.1f}"
        lines.append(
            f"{s.method:<14} {s.seed:<6d} "
            f"{s.identity_arcface:<7.3f} {s.outside_ssim:<7.3f} "
            f"{s.landmark_drift_px:<7.2f} "
            f"{s.dark_hole_fraction:<7.3f} {s.color_hue_shift:<7.2f} {crease_str:<7} "
            f"{s.cross_agreement:<7.3f} {s.tps_agreement:<7.3f} "
            f"{comp_str:<7} {'Y' if s.disqualified else 'N':<5}"
        )
    return "\n".join(lines)
