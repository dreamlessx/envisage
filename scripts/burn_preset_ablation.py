"""Cumulative preset ablation on a 21-case shared evaluation slice.

For 21 rhinoplasty cases from the shared evaluation pool, runs FLUX.Fill
at five preset-count levels (K=0, 1, 2, 4, 8) and scores each output via
compute_surgicalscore_v5. This is an in-domain saturation study with
substantial train overlap, not a held-out generalization result.

Preset ranking: GT-conditioned. `detect_rhino_changes(pre_lm, post_lm)`
returns per-preset delta magnitudes; we sort by magnitude (descending)
to pick top-K. For K=0 the output is the passthrough (input copy).

Outputs per case:
  evaluation/strengthen_v1/preset_ablation/<case>/k<K>/output.png
  evaluation/strengthen_v1/preset_ablation/<case>/k<K>/score.json

Aggregate:
  evaluation/strengthen_v1/preset_ablation.json
  paper/figures/preset_ablation_table.tex
  paper/figures/preset_ablation.pdf
  paper/figures/preset_ablation_summary.md

Usage:
  python3 scripts/burn_preset_ablation.py \\
    --test-split /data/.../hda_splits/test \\
    --output-dir evaluation/strengthen_v1/preset_ablation \\
    [--dry-run]   # skip FLUX calls, write dummy outputs for timing test
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

K_LEVELS = [0, 1, 2, 4, 8]
PROCEDURE = "rhinoplasty"
EXCLUDE = {"rhinoplasty_Nose_116"}
MATCHED_N = 20

# Priority order from rhino_config (most clinically impactful first).
# This is the tiebreaker when two presets have equal delta magnitude.
PRIORITY_ORDER = [
    "dorsal_hump_reduction",
    "tip_definition",
    "tip_narrowing",
    "dorsal_narrowing",
    "tip_rotation_up",
    "alar_base_narrowing",
    "dorsal_straightening",
    "nose_shortening",
]


# ---------------------------------------------------------------------------
# Preset ranking from GT landmark deltas
# ---------------------------------------------------------------------------

def rank_presets_by_delta(
    pre_lm, post_lm
) -> list[tuple[str, float]]:
    """Return presets sorted descending by GT-delta magnitude.

    Uses detect_rhino_changes to get raw deltas, maps each detected preset
    to its measurement delta, then sorts. Undetected presets get delta=0.0
    and appear last (priority order as tiebreaker).

    Returns list of (preset_key, magnitude) pairs, length 8.
    """
    from envisage.gt_analysis import detect_rhino_changes
    from envisage.rhino_config import RHINO_PROCEDURES

    detected_keys, sev, raw_deltas = detect_rhino_changes(pre_lm, post_lm)

    # Map preset -> delta magnitude. Use the measurement_key from RHINO_PROCEDURES
    # to look up the raw delta where possible; fall back to a small constant for
    # detected presets whose measurement key isn't directly in raw_deltas.
    MKEY_TO_DELTA: dict[str, str] = {
        "dorsal_hump_reduction": "bridge_spread",
        "dorsal_narrowing": "bridge_width_rel",
        "dorsal_straightening": "dorsal_deviation",
        "tip_narrowing": "tip_bulbosity",
        "tip_definition": "tip_bulbosity",
        "tip_rotation_up": "tip_vertical_shift",
        "alar_base_narrowing": "alar_width_rel",
        "nose_shortening": "nose_height_rel",
    }

    detected_set = set(detected_keys)

    scored: list[tuple[str, float, int]] = []  # (key, magnitude, priority)
    for priority, key in enumerate(PRIORITY_ORDER):
        delta_key = MKEY_TO_DELTA.get(key, "")
        raw = abs(raw_deltas.get(delta_key, 0.0)) if delta_key else 0.0
        if key not in detected_set:
            raw = 0.0
        scored.append((key, raw, priority))

    # Sort: detected first (raw > 0), then by magnitude desc, then priority asc
    scored.sort(key=lambda x: (-x[1], x[2]))

    return [(k, mag) for k, mag, _ in scored]


# ---------------------------------------------------------------------------
# Single inference call
# ---------------------------------------------------------------------------

def run_k(
    case_id: str,
    input_bgr: np.ndarray,
    target_bgr: np.ndarray,
    pre_lm,
    post_lm,
    ranked_presets: list[tuple[str, float]],
    k: int,
    out_dir: Path,
    pipe,
    *,
    seed: int = 42,
    steps: int = 28,
    guidance: float = 3.5,
    resolution: int = 1024,
    dry_run: bool = False,
) -> dict:
    """Run inference at preset level K, save output, return score dict."""
    from envisage.masks import generate_mask, MaskConfig
    from envisage.scorer import apply_hard_mask_composite
    from envisage.rhino_config import RhinoAnalysis, RHINO_PROCEDURES, Severity

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "output.png"

    # Build mask once (same for all K levels)
    mask = generate_mask(pre_lm, PROCEDURE, MaskConfig(dilation_px=25, feather_sigma=15))

    if k == 0:
        # Passthrough: output = input, no FLUX call
        output_bgr = input_bgr.copy()
        prompt_used = "PASSTHROUGH"
        presets_applied = []
    else:
        top_k_keys = [key for key, _ in ranked_presets[:k]]

        # Build RhinoAnalysis with only the top-K presets active
        detected_map = {key: (key in top_k_keys) for key in RHINO_PROCEDURES}
        sev_map = {key: Severity.MODERATE for key in top_k_keys}
        analysis = RhinoAnalysis(
            detected=detected_map,
            severity=sev_map,
            measurements={},
            level=1,
        )
        prompt = analysis.build_prompt(max_procedures=min(k, 3))
        prompt_used = prompt[:120] + "..." if len(prompt) > 120 else prompt
        presets_applied = top_k_keys

        if dry_run:
            # Write a copy of input as placeholder
            output_bgr = input_bgr.copy()
            log.info("[DRY-RUN] k=%d %s presets=%s", k, case_id, top_k_keys)
        else:
            import torch
            from PIL import Image as PILImage

            h_orig, w_orig = input_bgr.shape[:2]
            input_pil = PILImage.fromarray(
                cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)
            ).resize((resolution, resolution), PILImage.LANCZOS)

            mask_u8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
            mask_pil = PILImage.fromarray(mask_u8).resize(
                (resolution, resolution), PILImage.NEAREST
            )

            gen = torch.Generator(device="cpu").manual_seed(seed)
            t0 = time.perf_counter()
            result_pil = pipe(
                prompt=prompt,
                image=input_pil,
                mask_image=mask_pil,
                height=resolution,
                width=resolution,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=gen,
            ).images[0]
            elapsed = time.perf_counter() - t0

            result_bgr = cv2.cvtColor(
                np.array(result_pil.resize((w_orig, h_orig), PILImage.LANCZOS)),
                cv2.COLOR_RGB2BGR,
            )
            # Hard-mask composite: outside-mask pixels = input (architectural guarantee)
            output_bgr = apply_hard_mask_composite(result_bgr, input_bgr, mask)
            log.info("k=%d %s inference %.1fs", k, case_id, elapsed)

    # Save output
    cv2.imwrite(str(out_path), output_bgr)
    mask_u8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / "mask.png"), mask_u8)

    # Score via v5 scorer
    score = _score_one(case_id, input_bgr, target_bgr, output_bgr, mask, k)
    score["presets_applied"] = presets_applied
    score["prompt_used"] = prompt_used
    score["k"] = k

    (out_dir / "score.json").write_text(json.dumps(score, indent=2, default=_json_default))
    return score


# ---------------------------------------------------------------------------
# SurgicalScore v5 (inline, no subprocess)
# ---------------------------------------------------------------------------

_LPIPS_NET = None
_ARC_APP = None


def _lpips_net():
    global _LPIPS_NET
    if _LPIPS_NET is None:
        import lpips
        _LPIPS_NET = lpips.LPIPS(net="alex", verbose=False).eval()
    return _LPIPS_NET


def _arcface():
    global _ARC_APP
    if _ARC_APP is None:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(
            name="buffalo_l",
            root=str(Path.home() / ".insightface"),
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=-1, det_size=(320, 320))
        _ARC_APP = app
    return _ARC_APP


def _arc_emb(img_bgr):
    app = _arcface()
    faces = app.get(img_bgr)
    if not faces:
        return None
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    return np.asarray(f.embedding, dtype=np.float32)


def _cos(a, b):
    if a is None or b is None:
        return float("nan")
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _lpips(a_bgr, b_bgr):
    import torch
    net = _lpips_net()
    def to_t(x):
        rgb = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
    with torch.no_grad():
        return float(max(0.0, net(to_t(a_bgr), to_t(b_bgr)).item()))


def _rhino_morph(pts):
    """5-dim frontal rhino morphometry (mirrors compute_surgicalscore_v5)."""
    if pts is None or len(pts) < 478:
        return None
    L_INNER, R_INNER = 133, 362
    L_ALAR, R_ALAR = 64, 294
    TIP, NASION = 1, 168
    DORSUM_PTS = [168, 197, 195, 5, 4]
    inter = float(np.linalg.norm(pts[L_INNER] - pts[R_INNER]))
    if inter < 1.0:
        return None
    nasion, tip = pts[NASION], pts[TIP]
    alar = float(np.linalg.norm(pts[L_ALAR] - pts[R_ALAR])) / inter
    face_w = float(abs(pts[234, 0] - pts[454, 0]))
    nasal_face = float(np.linalg.norm(pts[L_ALAR] - pts[R_ALAR])) / max(face_w, 1.0)
    PHILT = 164
    mid_x = (pts[NASION, 0] + pts[PHILT, 0]) / 2 if PHILT < len(pts) else pts[NASION, 0]
    tip_sym = abs(tip[0] - mid_x) / inter
    line_dir = tip - nasion
    line_len = float(np.linalg.norm(line_dir))
    if line_len < 1.0:
        dorsum_rms = 0.0
    else:
        line_unit = line_dir / line_len
        normal = np.array([-line_unit[1], line_unit[0]])
        rms = [abs(float(np.dot(pts[i] - nasion, normal))) for i in DORSUM_PTS if i < len(pts)]
        dorsum_rms = float(np.sqrt(np.mean([r * r for r in rms]))) / line_len
    L_N, R_N = 79, 309
    if L_N < len(pts) and R_N < len(pts):
        l_h = float(abs(pts[L_N, 1] - pts[L_ALAR, 1]))
        r_h = float(abs(pts[R_N, 1] - pts[R_ALAR, 1]))
        nostril_asym = abs(l_h - r_h) / inter
    else:
        nostril_asym = 0.0
    return np.array([alar, nasal_face, tip_sym, dorsum_rms, nostril_asym], dtype=np.float64)


def _mam_perceptual(O, G, mask):
    ys, xs = np.where(mask > 127)
    if len(ys) < 100:
        return 0.5
    y0, y1 = ys.min(), ys.max(); x0, x1 = xs.min(), xs.max()
    y0 = max(0, y0 - 5); y1 = min(O.shape[0], y1 + 5)
    x0 = max(0, x0 - 5); x1 = min(O.shape[1], x1 + 5)
    Oc = cv2.resize(O[y0:y1, x0:x1], (256, 256))
    Gc = cv2.resize(G[y0:y1, x0:x1], (256, 256))
    return float(max(0.0, 1.0 - _lpips(Oc, Gc)))


def _realism(O):
    app = _arcface()
    faces = app.get(O)
    if not faces:
        return 0.0
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    return float(min(1.0, max(0.0, float(getattr(f, "det_score", 0.5)))))


def _outside_preserve(O, I, mask, tau=0.10):
    inv = 255 - mask
    ys, xs = np.where(inv > 127)
    if len(ys) < 100:
        return 1.0
    O_out, I_out = O.copy(), I.copy()
    inside = mask > 127
    O_out[inside] = 0; I_out[inside] = 0
    d = _lpips(O_out, I_out)
    return float(max(0.0, 1.0 - d / tau))


def _directional(d_O, d_G):
    no = float(np.linalg.norm(d_O)); ng = float(np.linalg.norm(d_G))
    eps = 1e-6
    if no < eps:
        return 0.0, 0.0
    c = max(min(float(np.dot(d_O, d_G) / (no * ng + eps)), 1.0), -1.0)
    A = (1.0 + c) / 2.0
    r = (no + eps) / (ng + eps)
    B = math.exp(-1.5 * max(0.0, math.log(r)) - 1.0 * max(0.0, -math.log(r)))
    return float(A), float(B)


def _score_one(case_id, I, G, O, mask_f32, k):
    """Compute SurgicalScore v5 for one (input, gt, output) triple."""
    from envisage.landmarks import extract_landmarks

    mask_u8 = (np.clip(mask_f32, 0, 1) * 255).astype(np.uint8)

    lm_I = extract_landmarks(I)
    lm_G = extract_landmarks(G)
    lm_O = extract_landmarks(O)

    if lm_I is None or lm_G is None or lm_O is None:
        return {"case": case_id, "k": k, "error": "landmark_failed", "SurgicalScore": float("nan")}

    pts_I, pts_G, pts_O = lm_I.points, lm_G.points, lm_O.points
    m_I = _rhino_morph(pts_I)
    m_G = _rhino_morph(pts_G)
    m_O = _rhino_morph(pts_O)

    if m_I is None or m_G is None or m_O is None:
        return {"case": case_id, "k": k, "error": "morph_failed", "SurgicalScore": float("nan")}

    h, w = I.shape[:2]
    d_O = m_O - m_I
    d_G = m_G - m_I
    A, B = _directional(d_O, d_G)
    A = max(0.0, min(1.0, A)); B = max(0.0, min(1.0, B))
    C = max(0.0, min(1.0, _mam_perceptual(O, G, mask_u8)))
    D = max(0.0, min(1.0, _realism(O)))
    E = max(0.0, min(1.0, _outside_preserve(O, I, mask_u8)))

    Raw_O = 0.40 * A + 0.30 * B + 0.15 * C + 0.10 * D + 0.05 * E

    # Passthrough calibration anchor
    A_I = B_I = 0.0
    C_I = max(0.0, min(1.0, _mam_perceptual(I, G, mask_u8)))
    D_I = max(0.0, min(1.0, _realism(I)))
    E_I = 1.0
    Raw_I = 0.40 * A_I + 0.30 * B_I + 0.15 * C_I + 0.10 * D_I + 0.05 * E_I

    emb_I = _arc_emb(I); emb_O = _arc_emb(O)
    arc_io = _cos(emb_I, emb_O)
    gate_pass = (not math.isnan(arc_io)) and (arc_io >= 0.65)

    invalid_cal = (1.0 - Raw_I) < 0.25
    if invalid_cal:
        SS = float("nan")
    else:
        SS_uncal = 0.30 + 0.70 * (Raw_O - Raw_I) / (1.0 - Raw_I)
        SS = SS_uncal if gate_pass else 0.0

    if math.isnan(SS):
        verdict = "INVALID"
    elif not gate_pass:
        verdict = "GATE_FAIL"
    elif SS >= 0.35:
        verdict = "PASS"
    elif SS >= 0.30:
        verdict = "BORDERLINE"
    else:
        verdict = "FAIL"

    return {
        "case": case_id, "k": k,
        "A": A, "B": B, "C": C, "D": D, "E": E,
        "Raw_O": Raw_O, "Raw_I": Raw_I,
        "arcface_io": arc_io, "gate_pass": bool(gate_pass),
        "SurgicalScore": SS,
        "invalid_calibration": invalid_cal,
        "verdict": verdict,
    }


def _json_default(o):
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o).__name__)


# ---------------------------------------------------------------------------
# Aggregate + artifacts
# ---------------------------------------------------------------------------

def aggregate(all_scores: list[dict]) -> dict:
    """Build aggregate table keyed by K."""
    by_k: dict[int, list[float]] = {k: [] for k in K_LEVELS}
    per_case: list[dict] = []

    for row in all_scores:
        if "error" in row:
            continue
        k = row["k"]
        ss = row.get("SurgicalScore", float("nan"))
        if not math.isnan(ss):
            by_k[k].append(ss)
        per_case.append(row)

    agg_mean, agg_std, agg_n = {}, {}, {}
    n_pass_035: dict[int, int] = {}
    for k in K_LEVELS:
        vals = [v for v in by_k[k] if not math.isnan(v)]
        agg_mean[k] = statistics.mean(vals) if vals else float("nan")
        agg_std[k] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        agg_n[k] = len(vals)
        n_pass_035[k] = sum(1 for v in vals if v >= 0.35)

    # Deltas vs K=0 and K=8
    mean0 = agg_mean.get(0, float("nan"))
    mean8 = agg_mean.get(8, float("nan"))

    delta_vs_passthrough = {}
    delta_vs_full = {}
    for k in K_LEVELS:
        mk = agg_mean[k]
        delta_vs_passthrough[k] = mk - mean0 if not (math.isnan(mk) or math.isnan(mean0)) else float("nan")
        delta_vs_full[k] = mk - mean8 if not (math.isnan(mk) or math.isnan(mean8)) else float("nan")

    return {
        "K": K_LEVELS,
        "per_case": per_case,
        "aggregate": {
            "mean": {k: agg_mean[k] for k in K_LEVELS},
            "std": {k: agg_std[k] for k in K_LEVELS},
            "n_valid": {k: agg_n[k] for k in K_LEVELS},
            "n_pass_035": {k: n_pass_035[k] for k in K_LEVELS},
            "delta_vs_passthrough": {k: delta_vs_passthrough[k] for k in K_LEVELS},
            "delta_vs_full": {k: delta_vs_full[k] for k in K_LEVELS},
        },
    }


def write_latex_table(agg: dict, out_path: Path) -> None:
    """Write preset_ablation_table.tex."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ag = _normalize_aggregate_keys(agg["aggregate"])
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{ccccccc}",
        r"\toprule",
        r"$K$ & Mean SS & Std & $n_\mathrm{pass}$/21 & $\Delta$ vs $K$=0 & $\Delta$ vs $K$=8 \\",
        r"\midrule",
    ]
    for k in K_LEVELS:
        mean = ag["mean"][k]
        std = ag["std"][k]
        n_p = ag["n_pass_035"][k]
        d0 = ag["delta_vs_passthrough"][k]
        d8 = ag["delta_vs_full"][k]
        def fmt(v):
            return f"{v:.3f}" if not math.isnan(v) else r"\textemdash"
        sign0 = "+" if (not math.isnan(d0) and d0 >= 0) else ""
        sign8 = "+" if (not math.isnan(d8) and d8 >= 0) else ""
        lines.append(
            f"{k} & {fmt(mean)} & {fmt(std)} & {n_p}/21 & "
            f"{sign0}{fmt(d0)} & {sign8}{fmt(d8)} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Cumulative preset ablation on the 21 rhinoplasty cases of the",
        r"shared evaluation pool (Section~\ref{sec:dataset}; 19 of 21 cases overlap",
        r"train, so this is an in-domain saturation curve, not held-out",
        r"generalization). SurgicalScore aggregated per preset-count level $K$,",
        r"with presets ordered per case by GT-conditioned landmark-delta magnitude",
        r"(oracle ranking; $K{=}1$ upper-bounds the deployable single-best",
        r"selector). $K{=}0$: passthrough anchor. $K{=}8$: full eight-preset bank.}",
        r"\label{tab:preset_ablation}",
        r"\end{table}",
    ]
    out_path.write_text("\n".join(lines) + "\n")
    log.info("LaTeX table: %s", out_path)


def write_plot(agg: dict, out_path: Path) -> None:
    """Write preset_ablation.pdf."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available; skipping PDF plot")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ag = _normalize_aggregate_keys(agg["aggregate"])
    ks = K_LEVELS
    means = [ag["mean"][k] for k in ks]
    stds = [ag["std"][k] for k in ks]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        ks, means, yerr=stds, fmt="o-", color="#2563EB", linewidth=2,
        markersize=7, capsize=4, label="Mean SurgicalScore v5",
    )
    ax.axhline(y=ag["mean"][0], color="#6B7280", linestyle="--", linewidth=1.2,
               label=f"Passthrough (K=0): {ag['mean'][0]:.3f}")
    ax.axhline(y=0.35, color="#EF4444", linestyle=":", linewidth=1.2,
               label="Pass threshold (0.35)")
    ax.set_xlabel("Preset count K", fontsize=12)
    ax.set_ylabel("Mean SurgicalScore v5", fontsize=12)
    ax.set_title("Cumulative Preset Ablation: Rhinoplasty (N=21)", fontsize=12)
    ax.set_xticks(ks)
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_path), format="pdf", dpi=150)
    plt.close()
    log.info("PDF plot: %s", out_path)


def write_summary(agg: dict, out_path: Path) -> None:
    """Write 1-paragraph summary for §5 splice."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate = agg["aggregate"] if "aggregate" in agg else agg
    ag = _normalize_aggregate_keys(aggregate)

    means = [ag["mean"][k] for k in K_LEVELS]
    stds = [ag["std"][k] for k in K_LEVELS]

    def fmt(v):
        return f"{v:.3f}" if not math.isnan(v) else "N/A"

    # Characterize the curve shape
    valid_means = [(k, m) for k, m in zip(K_LEVELS, means) if not math.isnan(m)]
    is_monotonic = all(valid_means[i][1] <= valid_means[i+1][1]
                       for i in range(len(valid_means) - 1))
    k4_mean = ag["mean"][4]; k8_mean = ag["mean"][8]
    saturation_gap = abs(k4_mean - k8_mean) if not (math.isnan(k4_mean) or math.isnan(k8_mean)) else float("nan")
    saturates_early = (not math.isnan(saturation_gap)) and (saturation_gap < 0.02)

    if is_monotonic and not saturates_early:
        shape = "monotonically improving"
        narrative = ("Each additional preset contributes measurable SurgicalScore improvement, "
                     "with no saturation plateau through K=8.")
    elif is_monotonic and saturates_early:
        shape = "monotonic with saturation at K=4"
        narrative = (f"The curve rises monotonically through K=4 (mean={fmt(k4_mean)}) "
                     f"and saturates between K=4 and K=8 (mean={fmt(k8_mean)}, "
                     f"gap={saturation_gap:.3f}). Four presets capture most of the "
                     f"signal; the remaining four refine without major lift.")
    else:
        shape = "non-monotonic"
        narrative = ("Honest read: preset ordering "
                     "by GT-delta magnitude does not produce strictly additive SurgicalScore "
                     "improvement. Review per-case scores to diagnose.")

    # K=0 anchor vs K=8
    d08 = ag["delta_vs_passthrough"][8]
    d08_str = f"+{d08:.3f}" if not math.isnan(d08) and d08 >= 0 else (f"{d08:.3f}" if not math.isnan(d08) else "N/A")

    n_pass_8 = ag["n_pass_035"][8]

    para = (
        f"We evaluated the contribution of each rhinoplasty preset by running FLUX.Fill "
        f"at five cumulative preset-count levels (K=0, 1, 2, 4, 8) across 21 rhinoplasty "
        f"cases from the shared evaluation pool (19 of 21 overlap train), scoring each "
        f"output with SurgicalScore v5. "
        f"At K=0 (passthrough), mean SurgicalScore was {fmt(ag['mean'][0])} "
        f"(std={fmt(ag['std'][0])}), establishing the calibration anchor. "
        f"At K=8 (all presets), mean SurgicalScore was {fmt(ag['mean'][8])} "
        f"(std={fmt(ag['std'][8])}), representing a {d08_str} absolute lift over passthrough, "
        f"with {n_pass_8}/21 cases exceeding the pass threshold (>=0.35). "
        f"The ablation curve was {shape}. "
        f"{narrative} "
        f"Because preset ordering is GT-conditioned, this summary should be read as an "
        f"oracle-style in-domain saturation analysis, not a deployable preset selector "
        f"or held-out generalization result."
    )

    out_path.write_text(para + "\n")
    log.info("Summary: %s", out_path)


def _normalize_aggregate_keys(aggregate: dict) -> dict:
    """Allow artifact writers to consume either live dicts or JSON-loaded payloads."""
    normalized: dict[str, dict[int, float | int]] = {}
    for field, values in aggregate.items():
        if isinstance(values, dict):
            normalized[field] = {int(key): value for key, value in values.items()}
        else:
            normalized[field] = values
    return normalized


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Preset ablation: rhino-only, K=0,1,2,4,8")
    ap.add_argument(
        "--test-split",
        required=True,
        help="Path to HDA test split root (flat layout with rhinoplasty_*_input.png)",
    )
    ap.add_argument(
        "--output-dir",
        default=REPO_ROOT / "evaluation" / "strengthen_v1" / "preset_ablation",
        type=Path,
        help="Per-case output root",
    )
    ap.add_argument(
        "--json-out",
        default=REPO_ROOT / "evaluation" / "strengthen_v1" / "preset_ablation.json",
        type=Path,
    )
    ap.add_argument(
        "--table-out",
        default=REPO_ROOT / "paper" / "figures" / "preset_ablation_table.tex",
        type=Path,
    )
    ap.add_argument(
        "--plot-out",
        default=REPO_ROOT / "paper" / "figures" / "preset_ablation.pdf",
        type=Path,
    )
    ap.add_argument(
        "--summary-out",
        default=REPO_ROOT / "paper" / "figures" / "preset_ablation_summary.md",
        type=Path,
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--dry-run", action="store_true", help="Skip FLUX calls; test scaffolding")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    test_root = Path(args.test_split)
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    # Discover N=21 matched rhino cases (flat layout)
    inputs = sorted(test_root.glob("rhinoplasty_*_input.png"))
    cases = []
    for p in inputs:
        stem = p.name.removesuffix("_input.png")
        if stem in EXCLUDE:
            log.info("Excluding %s", stem)
            continue
        tgt = p.with_name(f"{stem}_target.png")
        if tgt.exists():
            cases.append(stem)
    cases = cases[:MATCHED_N]
    log.info("Found %d matched rhino cases", len(cases))
    assert len(cases) == MATCHED_N, f"Expected {MATCHED_N}, found {len(cases)}"

    # Load models (skip for dry-run)
    pipe = None
    if not args.dry_run:
        import torch
        from diffusers import FluxFillPipeline

        token = os.environ.get("HF_TOKEN")
        if not token:
            env_file = REPO_ROOT / ".env"
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    if line.startswith("HF_TOKEN="):
                        token = line.split("=", 1)[1].strip()
                        break
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        log.info("Loading FLUX.1-Fill-dev on %s...", device)
        pipe = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            torch_dtype=dtype,
            token=token,
        )
        if device == "cuda":
            pipe = pipe.to(device)
        else:
            pipe.enable_model_cpu_offload()
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass
        pipe.set_progress_bar_config(disable=True)

    from envisage.landmarks import extract_landmarks

    all_scores: list[dict] = []
    t_total = time.perf_counter()

    for case_id in cases:
        input_path = test_root / f"{case_id}_input.png"
        target_path = test_root / f"{case_id}_target.png"

        input_bgr = cv2.imread(str(input_path))
        target_bgr = cv2.imread(str(target_path))
        if input_bgr is None or target_bgr is None:
            log.error("Cannot read images for %s; skipping", case_id)
            continue

        pre_lm = extract_landmarks(input_bgr)
        post_lm = extract_landmarks(target_bgr)
        if pre_lm is None or post_lm is None:
            log.error("Landmark extraction failed for %s; skipping", case_id)
            continue

        ranked = rank_presets_by_delta(pre_lm, post_lm)
        log.info(
            "Case %s: ranked presets %s",
            case_id,
            [(k, f"{m:.3f}") for k, m in ranked],
        )

        for k in K_LEVELS:
            out_dir = output_root / case_id / f"k{k}"
            score_file = out_dir / "score.json"

            # Resume: skip if already scored
            if score_file.exists():
                try:
                    existing = json.loads(score_file.read_text())
                    if "SurgicalScore" in existing and not existing.get("error"):
                        log.info("Skipping %s k=%d (cached)", case_id, k)
                        all_scores.append(existing)
                        continue
                except Exception:
                    pass

            score = run_k(
                case_id=case_id,
                input_bgr=input_bgr,
                target_bgr=target_bgr,
                pre_lm=pre_lm,
                post_lm=post_lm,
                ranked_presets=ranked,
                k=k,
                out_dir=out_dir,
                pipe=pipe,
                seed=args.seed,
                steps=args.steps,
                guidance=args.guidance,
                resolution=args.resolution,
                dry_run=args.dry_run,
            )
            all_scores.append(score)
            ss_str = f"{score.get('SurgicalScore', 'nan'):.3f}" if isinstance(
                score.get("SurgicalScore"), float) else "nan"
            log.info(
                "  %s k=%d SS=%s verdict=%s presets=%s",
                case_id, k, ss_str,
                score.get("verdict", "?"),
                score.get("presets_applied", []),
            )

    elapsed = time.perf_counter() - t_total
    log.info("Done: %d scores in %.1f s", len(all_scores), elapsed)

    # Aggregate
    agg = aggregate(all_scores)
    agg["elapsed_s"] = elapsed
    agg["n_cases"] = len(cases)

    # Print summary table to stdout
    print("\n=== PRESET ABLATION RESULTS ===")
    print(f"{'K':>4}  {'Mean SS':>8}  {'Std':>6}  {'n_pass/21':>10}  {'vs K=0':>8}  {'vs K=8':>8}")
    ag = agg["aggregate"]
    for k in K_LEVELS:
        m = ag["mean"][k]; s = ag["std"][k]
        np35 = ag["n_pass_035"][k]
        d0 = ag["delta_vs_passthrough"][k]
        d8 = ag["delta_vs_full"][k]
        def fmt(v):
            return f"{v:+.3f}" if not math.isnan(v) else "  N/A "
        print(f"{k:>4}  {m:8.3f}  {s:6.3f}  {np35:>4}/21      {fmt(d0):>8}  {fmt(d8):>8}")

    # Write artifacts
    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(agg, indent=2, default=_json_default))
    log.info("JSON: %s", json_out)

    write_latex_table(agg, Path(args.table_out))
    write_plot(agg, Path(args.plot_out))
    write_summary(agg, Path(args.summary_out))

    print(f"\nArtifacts:")
    print(f"  {args.json_out}")
    print(f"  {args.table_out}")
    print(f"  {args.plot_out}")
    print(f"  {args.summary_out}")


if __name__ == "__main__":
    main()
