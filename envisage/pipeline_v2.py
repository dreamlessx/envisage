"""Ensemble pipeline v2: preset-driven multi-method decision.

End-to-end entry point that composes the new Phase 1 components:

    input_bgr + procedure
         │
         ▼
   prepare_context  -> landmarks, mask, preset_prompt, warped, depth_modified
         │
         ▼
   generate_candidates  -> 25 Candidates (M1-M5 x seeds)
         │
         ▼
   select_best  -> winner Candidate + CandidateScore + all_scores
         │
         ▼
   EnsembleResult

If the scorer disqualifies all diffusion candidates, the TPS fallback
(M5) is returned -- the zero-hallucination guarantee.

Run this module as a script to smoke-test on a single image:

    .venv/bin/python -m envisage.pipeline_v2 --image <path> --procedure rhinoplasty

which runs M1 + M5 only (M2-M4 still stubbed) and prints the candidate
score table.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .candidates import (
    CandidateContext,
    DEFAULT_SEEDS,
    enabled_methods,
    generate_candidates,
    prepare_context,
)
from .scorer import (
    Candidate,
    CandidateScore,
    apply_hard_mask_composite,
    format_scores,
    select_best,
)

log = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    """Everything produced by a single pipeline_v2 run."""

    input_bgr: np.ndarray
    procedure: str
    output_bgr: np.ndarray          # the shipped output
    winner: Candidate               # which candidate was selected
    winner_score: CandidateScore    # its score decomposition
    all_scores: list[CandidateScore]
    fallback_activated: bool        # True when the TPS fallback won
    timing_s: dict[str, float] = field(default_factory=dict)
    analysis: Any | None = None     # the Analysis object used for scoring

    def explain(self) -> str:
        """Human-readable summary of what was produced and why.

        Intended for the final displayed output next to each image, so
        viewers see which method won, which presets were applied, and
        whether the fallback activated. This is the requirement that
        'when displaying a final image, explain what actually got
        presented'.
        """
        lines: list[str] = []
        # What got shipped
        if self.fallback_activated:
            lines.append(
                f"OUTPUT: TPS FALLBACK (zero-hallucination guarantee). "
                f"All {sum(1 for s in self.all_scores if s.method != self.winner.method)} "
                f"diffusion candidates disqualified."
            )
        else:
            lines.append(
                f"OUTPUT: {self.winner.method} seed={self.winner.seed} "
                f"composite={self.winner_score.composite:.3f}"
            )

        # Which presets were applied (if an Analysis is available)
        if self.analysis is not None:
            active = list(getattr(self.analysis, "active_keys", []) or [])
            if active:
                sev_map = getattr(self.analysis, "severity", {}) or {}
                sev_labels = {0: "NONE", 1: "MILD", 2: "MOD", 3: "SEV"}
                parts = []
                for k in active:
                    sev = sev_map.get(k, 2)
                    parts.append(f"{k} ({sev_labels.get(sev, '?')})")
                lines.append(f"PRESETS APPLIED: {', '.join(parts)}")
            else:
                lines.append("PRESETS APPLIED: none")

        # Fidelity summary on the winner
        if self.winner_score.fidelity_checked:
            if self.winner_score.fidelity_passes:
                lines.append("FIDELITY: all active presets within severity band; inactive presets within drift tolerance")
            else:
                lines.append(
                    f"FIDELITY: failures = {list(self.winner_score.fidelity_failures)}"
                )

        # Disqualify summary across the pool
        n_disq = sum(1 for s in self.all_scores if s.disqualified)
        n_total = len(self.all_scores)
        lines.append(f"CANDIDATE POOL: {n_total - n_disq}/{n_total} survived gates")

        return "\n".join(lines)


def summarize_fallback_rate(
    results: list[EnsembleResult],
    warn_threshold: float = 0.30,
) -> dict[str, Any]:
    """Aggregate fallback-activation rate across a batch of runs.

    Intended for pipeline-health monitoring. If the TPS fallback gets
    shipped too often, it is a signal that the diffusion methods are
    failing their gates systematically -- either the gates are too tight,
    the LoRAs are undertrained, or the prompt composition is off.

    Args:
        results: list of EnsembleResults from a test-set evaluation.
        warn_threshold: fraction above which we flag a procedure as
            unhealthy (default 30%).

    Returns:
        Dict with overall + per-procedure fallback rates and a list of
        procedures that exceeded the warn threshold.
    """
    if not results:
        return {"total": 0, "overall_rate": 0.0, "by_procedure": {}, "warnings": []}

    by_proc: dict[str, dict[str, int]] = {}
    for r in results:
        bucket = by_proc.setdefault(r.procedure, {"total": 0, "fallback": 0})
        bucket["total"] += 1
        if r.fallback_activated:
            bucket["fallback"] += 1

    by_proc_rates = {
        proc: {
            "total": bucket["total"],
            "fallback": bucket["fallback"],
            "rate": bucket["fallback"] / max(bucket["total"], 1),
        }
        for proc, bucket in by_proc.items()
    }

    warnings = [
        proc for proc, r in by_proc_rates.items()
        if r["rate"] > warn_threshold
    ]

    total = sum(b["total"] for b in by_proc.values())
    fallback = sum(b["fallback"] for b in by_proc.values())
    return {
        "total": total,
        "overall_rate": fallback / max(total, 1),
        "by_procedure": by_proc_rates,
        "warnings": warnings,
        "warn_threshold": warn_threshold,
    }


def run_ensemble(
    input_bgr: np.ndarray,
    procedure: str,
    *,
    pipe: Any | None = None,
    has_controlnet: bool = False,
    depth_estimator: Any | None = None,
    intensity_pct: float = 50.0,
    preset_prompt: str | None = None,
    seeds: list[int] | None = None,
    methods: list[str] | None = None,
) -> EnsembleResult:
    """Run the decision pipeline end-to-end on one input.

    Args:
        input_bgr: Pre-op BGR image (OpenCV convention).
        procedure: "rhinoplasty" | "blepharoplasty" | "rhytidectomy" | "orthognathic".
        pipe, has_controlnet, depth_estimator: Passed to `prepare_context`.
            Pass None for a TPS-only run (useful for CPU smoke testing).
        intensity_pct: 0-100 severity scaling for diffusion + depth.
        preset_prompt: Override the prompt. Default uses build_adaptive_prompt.
        seeds: Override DEFAULT_SEEDS.
        methods: Override enabled methods.

    Returns:
        EnsembleResult with all candidates scored and the winner selected.

    Raises:
        ValueError: if no face is detected or the context cannot be built.
    """
    timing: dict[str, float] = {}

    t0 = time.perf_counter()
    ctx = prepare_context(
        input_bgr=input_bgr,
        procedure=procedure,
        pipe=pipe,
        has_controlnet=has_controlnet,
        depth_estimator=depth_estimator,
        intensity_pct=intensity_pct,
        preset_prompt=preset_prompt,
    )
    timing["prepare_context"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    raw_cands = generate_candidates(ctx, seeds=seeds, methods=methods)
    timing["generate_candidates"] = time.perf_counter() - t0

    # Hard-mask composite: replace each candidate's image with a mask-alpha
    # blend of prediction and input. Outside-mask pixels become byte-
    # identical to input (architectural identity guarantee, pushes outside
    # SSIM to 1.0 instead of the soft-blended ~0.98). The TPS fallback is
    # already outside-mask-identical by construction; compositing is a
    # no-op on it but we apply uniformly for consistency.
    t0 = time.perf_counter()
    cands = []
    for c in raw_cands:
        composited = apply_hard_mask_composite(c.image_bgr, ctx.input_bgr, ctx.mask)
        cands.append(Candidate(
            image_bgr=composited,
            method=c.method,
            seed=c.seed,
            metadata={**c.metadata, "hard_composite": True},
        ))
    timing["hard_composite"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    winner, win_score, all_scores = select_best(
        cands, ctx.input_bgr, ctx.mask,
        procedure=procedure, analysis=ctx.analysis,
    )
    timing["select_best"] = time.perf_counter() - t0

    # Per-candidate trace: log each method/seed combo's gate results.
    # Essential for diagnosing why a diffusion method loses to TPS.
    for s in all_scores:
        if s.disqualified:
            log.info(
                "CAND %s seed=%d DQ: %s",
                s.method, s.seed, s.disqualify_reasons,
            )
        else:
            log.info(
                "CAND %s seed=%d OK composite=%.3f arc=%.3f out=%.3f",
                s.method, s.seed, s.composite,
                s.identity_arcface, s.outside_ssim,
            )

    fallback = winner.is_fallback
    if fallback:
        log.warning(
            "Ensemble shipped TPS fallback for procedure=%s "
            "(all %d diffusion candidates disqualified)",
            procedure, sum(1 for c in cands if not c.is_fallback),
        )
    else:
        log.info(
            "Ensemble shipped %s seed=%d composite=%.3f for procedure=%s",
            winner.method, winner.seed, win_score.composite, procedure,
        )

    return EnsembleResult(
        input_bgr=input_bgr,
        procedure=procedure,
        output_bgr=winner.image_bgr,
        winner=winner,
        winner_score=win_score,
        all_scores=all_scores,
        fallback_activated=fallback,
        timing_s=timing,
        analysis=ctx.analysis,
    )


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Envisage ensemble pipeline v2 CLI")
    parser.add_argument("--image", required=True, help="Path to input BGR image")
    parser.add_argument(
        "--procedure",
        required=True,
        choices=["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic"],
    )
    parser.add_argument("--output", default=None, help="Where to save the winner output")
    parser.add_argument("--intensity", type=float, default=50.0, help="0-100 severity")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Override method keys (default: all enabled). Use 'M5_tps' alone for TPS-only.",
    )
    parser.add_argument(
        "--no-diffusion",
        action="store_true",
        help="Skip loading FLUX; run TPS-only (fast smoke test).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    img = cv2.imread(args.image)
    if img is None:
        print(f"ERROR: cannot read image {args.image}")
        return 1

    pipe = None
    has_controlnet = False
    depth_estimator = None

    if not args.no_diffusion:
        try:
            # scripts/run_production.py is the canonical FLUX loader.
            # Add scripts/ to path so we can import it here.
            import sys
            from pathlib import Path as _P
            scripts_dir = _P(__file__).resolve().parent.parent / "scripts"
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
            from run_production import load_flux_pipeline  # type: ignore

            from .depth import DepthEstimator
            pipe = load_flux_pipeline(use_controlnet=True)
            has_controlnet = True
            depth_estimator = DepthEstimator()
        except Exception as e:
            log.error("Failed to load FLUX pipeline: %s. Falling back to TPS-only.", e)
            pipe = None

    methods = args.methods or (["M5_tps"] if args.no_diffusion or pipe is None else None)

    try:
        result = run_ensemble(
            input_bgr=img,
            procedure=args.procedure,
            pipe=pipe,
            has_controlnet=has_controlnet,
            depth_estimator=depth_estimator,
            intensity_pct=args.intensity,
            methods=methods,
        )
    except Exception as e:
        log.exception("Ensemble run failed: %s", e)
        return 2

    print()
    print(format_scores(result.all_scores))
    print()
    print(f"WINNER:  {result.winner.method}  seed={result.winner.seed}  "
          f"composite={result.winner_score.composite:.3f}  "
          f"fallback={result.fallback_activated}")
    print(f"TIMING:  {result.timing_s}")

    if args.output:
        cv2.imwrite(args.output, result.output_bgr)
        print(f"Output saved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
