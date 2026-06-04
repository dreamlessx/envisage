"""Multi-method candidate generation for the decision pipeline.

Five methods x K seeds per case:
  M1  FLUX + depth ControlNet  (pretrained, wired)
  M2  FLUX.1-Fill + ICEdit MoE-LoRA         (stub; awaits training)
  M3  FLUX.1-Kontext + paired LoRA          (stub; awaits training)
  M4  FLUX.1-Fill + mask-aware LoRA         (stub; awaits training)
  M5  TPS warp + depth modify  (no diffusion, deterministic, always-on)

The TPS method is the zero-hallucination fallback. It is deterministic
given the same landmarks and parameters, so only one seed is evaluated.

Shared expensive-to-compute artifacts (pipe, landmarks, mask, depth,
prompt, TPS-warped PIL) are computed once by `prepare_context()` and
passed to every method via `CandidateContext`.

This module's job is ONLY to orchestrate generation. Scoring and
best-candidate selection live in `envisage/scorer.py`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np

from .landmarks import FaceLandmarks
from .scorer import Candidate, TPS_METHOD

log = logging.getLogger(__name__)


DEFAULT_SEEDS: list[int] = [42, 123, 456, 789, 1024]


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Method:
    key: str
    label: str
    is_trained: bool
    is_diffusion: bool
    enabled: bool


METHODS: dict[str, Method] = {
    "M1_flux_zs": Method(
        key="M1_flux_zs",
        label="FLUX + depth ControlNet (pretrained)",
        is_trained=False,
        is_diffusion=True,
        enabled=True,
    ),
    "M2_icedit": Method(
        key="M2_icedit",
        label="FLUX.1-Fill + ICEdit MoE-LoRA",
        is_trained=True,
        is_diffusion=True,
        enabled=False,
    ),
    "M3_kontext": Method(
        key="M3_kontext",
        label="FLUX.1-Kontext + paired LoRA",
        is_trained=True,
        is_diffusion=True,
        enabled=False,
    ),
    "M4_fill_lora": Method(
        key="M4_fill_lora",
        label="FLUX.1-Fill + mask-aware LoRA",
        is_trained=True,
        is_diffusion=True,
        enabled=False,
    ),
    TPS_METHOD: Method(
        key=TPS_METHOD,
        label="TPS warp + depth modify (deterministic fallback)",
        is_trained=False,
        is_diffusion=False,
        enabled=True,
    ),
}


# ---------------------------------------------------------------------------
# Shared context (built once per input, reused across methods + seeds)
# ---------------------------------------------------------------------------

@dataclass
class CandidateContext:
    """All shared inputs needed by the ensemble methods.

    Built once per input via `prepare_context()`, then reused across
    methods and seeds. Avoids redundant landmark extraction, mask
    generation, depth estimation, and TPS warping.
    """

    input_bgr: np.ndarray
    procedure: str
    landmarks: FaceLandmarks
    mask: np.ndarray                   # float32 [0,1], surgical region
    preset_prompt: str                 # pre-built prompt from preset analyzer
    intensity_pct: float               # 0-100 diffusion strength scaling

    # Optional pre-computed artifacts (populated by prepare_context)
    warped_bgr: np.ndarray | None = None          # TPS-warped pre-image
    warped_pil: Any | None = None                 # PIL version of warped_bgr
    depth_modified: np.ndarray | None = None      # anatomical depth map

    # Diffusion runtime (required for M1-M4; None-OK for M5-only runs)
    pipe: Any | None = None
    has_controlnet: bool = False

    # Inpainting strength computed per-procedure
    inpainting_strength: float = 0.65

    # Procedure-specific Analysis object (RhinoAnalysis / BlephAnalysis /
    # RhytidAnalysis). Passed through to the scorer's fidelity gate.
    analysis: Any | None = None

    extra: dict[str, Any] = field(default_factory=dict)


def prepare_context(
    input_bgr: np.ndarray,
    procedure: str,
    *,
    pipe: Any | None = None,
    has_controlnet: bool = False,
    depth_estimator: Any | None = None,
    intensity_pct: float = 50.0,
    preset_prompt: str | None = None,
) -> CandidateContext:
    """Build the shared context. Does the landmark / mask / depth / TPS work once.

    If `pipe` is None, the context is usable only for M5 (TPS) candidates.
    Callers running diffusion methods must pass a loaded pipe.
    """
    from PIL import Image
    from .depth import modify_depth
    from .landmarks import extract_landmarks
    from .masks import (
        MaskConfig, active_bleph_mask_config, active_rhyt_mask_config,
        generate_adaptive_bleph_mask,
        generate_adaptive_rhytid_mask, generate_mask,
    )
    from .pipeline import (
        apply_surgical_tps_warp,
        build_adaptive_prompt,
    )

    landmarks = extract_landmarks(input_bgr)
    if landmarks is None:
        raise ValueError("No face detected in input")

    # Procedure-specific mask
    if procedure == "blepharoplasty":
        mask = generate_adaptive_bleph_mask(
            landmarks, active_bleph_mask_config(), intensity_pct,
        )
    elif procedure == "rhytidectomy":
        mask = generate_adaptive_rhytid_mask(
            landmarks, active_rhyt_mask_config(),
            include_neck=False,
        )
    else:
        mask = generate_mask(
            landmarks, procedure, MaskConfig(dilation_px=25, feather_sigma=15),
        )

    # TPS pre-warp (also serves as M5 base)
    try:
        warped_bgr = apply_surgical_tps_warp(input_bgr, landmarks, procedure)
    except Exception as e:
        log.warning("TPS pre-warp failed: %s", e)
        warped_bgr = input_bgr.copy()

    warped_pil = Image.fromarray(cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB))

    # Depth estimation + anatomical modification
    depth_modified = None
    if depth_estimator is not None:
        try:
            input_pil = Image.fromarray(cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB))
            depth_orig = depth_estimator.estimate(input_pil)
            depth_modified = modify_depth(
                depth_orig, landmarks, mask, procedure, intensity_pct=intensity_pct,
            )
        except Exception as e:
            log.warning("Depth estimation / modification failed: %s", e)

    # Analysis object for the fidelity gate. Uses the same per-procedure
    # analyzers that populate the prompt, so prompt intent and fidelity
    # intent stay aligned. Callers may override via `preset_prompt`, in
    # which case the analysis is still populated for scoring.
    analysis: Any | None = None
    try:
        if procedure == "rhinoplasty":
            from .rhino_config import analyze_rhinoplasty
            analysis = analyze_rhinoplasty(landmarks)
        elif procedure == "blepharoplasty":
            from .bleph_config import analyze_blepharoplasty
            analysis = analyze_blepharoplasty(landmarks)
        elif procedure == "rhytidectomy":
            from .rhytid_config import analyze_rhytidectomy
            analysis = analyze_rhytidectomy(landmarks)
    except Exception as e:
        log.warning("prepare_context: analyzer failed (%s); fidelity gate will be skipped", e)

    # Prompt: use preset analyzer if not supplied
    if preset_prompt is None:
        if analysis is not None and hasattr(analysis, "build_prompt"):
            preset_prompt = analysis.build_prompt(max_procedures=3)
        else:
            preset_prompt = build_adaptive_prompt(procedure, landmarks)

    # Per-procedure inpainting strength
    if procedure == "blepharoplasty":
        strength = 0.3 + 0.25 * (intensity_pct / 100.0)
    elif procedure == "rhytidectomy":
        strength = 0.65
    else:
        strength = 0.65 + 0.20 * (intensity_pct / 100.0)

    return CandidateContext(
        input_bgr=input_bgr,
        procedure=procedure,
        landmarks=landmarks,
        mask=mask,
        preset_prompt=preset_prompt,
        intensity_pct=intensity_pct,
        warped_bgr=warped_bgr,
        warped_pil=warped_pil,
        depth_modified=depth_modified,
        pipe=pipe,
        has_controlnet=has_controlnet,
        inpainting_strength=strength,
        analysis=analysis,
    )


# ---------------------------------------------------------------------------
# Individual method implementations
# ---------------------------------------------------------------------------

def _gen_m1_flux_zs(ctx: CandidateContext, seed: int) -> np.ndarray:
    """M1: current zero-shot FLUX + depth ControlNet pipeline, single seed.

    Mirrors the seed-sweep branch of `pipeline.run_pipeline` exactly,
    so M1 candidates reproduce the current paper baseline 1:1.
    """
    from .pipeline import run_single_seed

    if ctx.pipe is None:
        raise RuntimeError("M1 requires a loaded FLUX pipe in CandidateContext.pipe")
    if ctx.depth_modified is None:
        raise RuntimeError("M1 requires modified depth in CandidateContext.depth_modified")

    result_pil = run_single_seed(
        pipe=ctx.pipe,
        has_controlnet=ctx.has_controlnet,
        input_pil=ctx.warped_pil,
        mask=ctx.mask,
        modified_depth=ctx.depth_modified,
        prompt=ctx.preset_prompt,
        procedure=ctx.procedure,
        seed=seed,
        inpainting_strength=ctx.inpainting_strength,
    )
    return cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)


def _gen_m5_tps(ctx: CandidateContext, seed: int) -> np.ndarray:
    """M5: deterministic TPS warp + depth modification, no diffusion.

    Uses the already-computed warped_bgr from the context. Seed is
    ignored (deterministic given the same context).
    """
    # The pre-warp in prepare_context already did this work. If it
    # degenerated to a copy of the input (unknown procedure), we still
    # ship it -- that represents the "no-op baseline" correctly.
    return ctx.warped_bgr if ctx.warped_bgr is not None else ctx.input_bgr.copy()


# ---------------------------------------------------------------------------
# M2: ICEdit MoE-LoRA via pretrained sanaka87 weights (fast path, no training)
# ---------------------------------------------------------------------------

_M2_PIPE = None
_M2_LOAD_FAILED = False

ICEDIT_REPO = "sanaka87/ICEdit-MoE-LoRA"
ICEDIT_BASE_MODEL = "black-forest-labs/FLUX.1-Fill-dev"


def _load_m2_icedit_pipe() -> Any | None:
    """Lazy-load FLUX.1-Fill-dev + ICEdit MoE-LoRA.

    Cached singleton. Returns None if load fails (no GPU, no HF_TOKEN, no
    network). Callers treat None as "M2 unavailable"; the ensemble still
    has M1 and M5. Per arxiv:2504.20690 ICEdit is 4 experts, rank 32,
    TopK=1 on FLUX.1-Fill.
    """
    global _M2_PIPE, _M2_LOAD_FAILED

    if _M2_LOAD_FAILED:
        return None
    if _M2_PIPE is not None:
        return _M2_PIPE

    try:
        import os
        import torch
        from diffusers import FluxFillPipeline

        token = os.environ.get("HF_TOKEN")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        pipe = FluxFillPipeline.from_pretrained(
            ICEDIT_BASE_MODEL, torch_dtype=dtype, token=token,
        )
        # Keep the whole pipe on one device. cpu_offload + LoRA causes
        # "Input type CPUBFloat16 and weight type CUDABFloat16" mismatches
        # because offload hooks move the transformer but not the input
        # tensors built in the pipe's __call__ preprocessor.
        pipe = pipe.to(device)
        pipe.load_lora_weights(ICEDIT_REPO)
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass
        pipe.set_progress_bar_config(disable=True)

        _M2_PIPE = pipe
        log.info("M2 ICEdit (FLUX.1-Fill + sanaka87/ICEdit-MoE-LoRA) loaded on %s", device)
        return pipe
    except Exception as e:
        log.warning("M2 ICEdit load failed (%s); method will stay disabled", e)
        _M2_LOAD_FAILED = True
        return None


def _gen_m2_icedit(ctx: CandidateContext, seed: int) -> np.ndarray:
    """M2: FLUX.1-Fill + ICEdit MoE-LoRA inference.

    The ICEdit MoE-LoRA is pretrained (no Envisage-specific training
    required). We feed the same preset-composed prompt + mask + warped
    pre-image that M1 sees, so cross-method agreement on the ensemble
    reflects model differences rather than input divergence.
    """
    import torch
    from PIL import Image

    pipe = _load_m2_icedit_pipe()
    if pipe is None:
        raise RuntimeError(
            "M2 ICEdit pipeline unavailable; call enable_m2_icedit() first "
            "or set HF_TOKEN + ensure a GPU is present"
        )

    h, w = ctx.input_bgr.shape[:2]
    target = 1024  # ICEdit trained at 1024x1024
    size = (target, target)

    pil_img = ctx.warped_pil
    if pil_img is None:
        pil_img = Image.fromarray(cv2.cvtColor(ctx.warped_bgr or ctx.input_bgr,
                                                cv2.COLOR_BGR2RGB))
    pil_img = pil_img.resize(size, Image.LANCZOS)

    mask = ctx.mask
    if mask.shape[:2] != size:
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR)
    if mask.dtype == np.float32 or mask.dtype == np.float64:
        mask_u8 = np.clip(mask * 255, 0, 255).astype(np.uint8)
    else:
        mask_u8 = mask.astype(np.uint8)
    mask_pil = Image.fromarray(mask_u8)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    result = pipe(
        prompt=ctx.preset_prompt,
        image=pil_img,
        mask_image=mask_pil,
        height=target,
        width=target,
        strength=ctx.inpainting_strength,
        guidance_scale=3.5,
        num_inference_steps=20,
        generator=gen,
    )
    out_pil = result.images[0].resize((w, h), Image.LANCZOS)
    return cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)


def enable_m2_icedit() -> bool:
    """Attempt to load ICEdit and flip M2 on if successful.

    Call once at startup after confirming HF_TOKEN + GPU. Returns True
    if M2 is now enabled and ready for inference, False if load failed
    (in which case M2 stays disabled and the ensemble runs M1/M3/M4/M5
    only).
    """
    pipe = _load_m2_icedit_pipe()
    if pipe is None:
        return False
    # Mutate the registry entry to enabled=True
    meta = METHODS["M2_icedit"]
    METHODS["M2_icedit"] = Method(**{**meta.__dict__, "enabled": True})
    log.info("Method M2_icedit enabled (ICEdit-MoE-LoRA loaded)")
    return True


# ---------------------------------------------------------------------------
# M3: FLUX.1-Kontext + paired LoRA (trained on HDA pre-to-post)
# ---------------------------------------------------------------------------

KONTEXT_BASE_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"
KONTEXT_LORA_PATHS: dict[str, str] = {
    "rhinoplasty": "checkpoints/kontext_standalone/rhinoplasty/final",
    "blepharoplasty": "checkpoints/kontext_standalone/blepharoplasty/final",
    "rhytidectomy": "checkpoints/kontext_standalone/rhytidectomy/final",
}

_M3_PIPE = None                # shared FluxPipeline for Kontext
_M3_CURRENT_PROCEDURE: str | None = None  # which procedure's LoRA is currently loaded
_M3_LOAD_FAILED = False
# When True, load the PEFT adapter from checkpoints/kontext_standalone/*.
# When False, run FLUX.1-Kontext-dev zero-shot with no adapter attached.
# Default False aligns with enable_all_trained_methods(m3_use_lora=False):
# the trained Kontext LoRAs produce passthrough per iter #13 audit, whereas
# zero-shot (iter #39 probe, job 10181103) produced visibly modified outputs
# with outside_ssim 0.963-0.994 and arc(out,gt) 0.603 (bleph) / 0.427 (rhino).
# Flip to True via enable_m3_kontext(use_lora=True) when new LoRAs land.
_M3_USE_LORA: bool = False


def _resolve_checkpoint_path(rel_path: str) -> Any | None:
    """Resolve a checkpoint path. Returns a Path if found, else None."""
    from pathlib import Path

    p = Path(__file__).resolve().parent.parent / rel_path
    return p if p.exists() else None


def _load_m3_kontext_pipe(procedure: str) -> Any | None:
    """Load FLUX.1-Kontext-dev + procedure-specific PEFT LoRA.

    The Kontext LoRAs on disk were trained with `peft` and saved via
    `adapter_model.safetensors`. They are NOT in diffusers-native LoRA
    format, so we load them onto `pipe.transformer` via
    `PeftModel.from_pretrained`, matching the pattern in
    `scripts/run_production.load_flux_pipeline`.

    To swap procedures we unload the previous PEFT adapter (by reassigning
    the base transformer back) and load the next one. On first load the
    transformer is cached so subsequent swaps are fast.
    """
    global _M3_PIPE, _M3_CURRENT_PROCEDURE, _M3_LOAD_FAILED

    if _M3_LOAD_FAILED:
        return None

    lora_path = None
    if _M3_USE_LORA:
        lora_rel = KONTEXT_LORA_PATHS.get(procedure)
        if lora_rel is None:
            log.warning("M3: no Kontext LoRA registered for procedure %r", procedure)
            return None
        lora_path = _resolve_checkpoint_path(lora_rel)
        if lora_path is None:
            log.warning("M3: Kontext LoRA checkpoint not found at %s", lora_rel)
            return None

    try:
        import os
        import torch
        from peft import PeftModel
        # Kontext is image-to-image; plain FluxPipeline is text-to-image.
        # Try the dedicated FluxKontextPipeline if diffusers exposes it;
        # otherwise fall back to the img2img pipeline.
        try:
            from diffusers import FluxKontextPipeline as _KontextCls
        except ImportError:
            try:
                from diffusers import FluxImg2ImgPipeline as _KontextCls  # type: ignore
            except ImportError:
                from diffusers import FluxPipeline as _KontextCls  # type: ignore

        if _M3_PIPE is None:
            token = os.environ.get("HF_TOKEN")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            _M3_PIPE = _KontextCls.from_pretrained(
                KONTEXT_BASE_MODEL, torch_dtype=dtype, token=token,
            )
            # Move to device BEFORE LoRA, not after. cpu_offload + LoRA
            # breaks with CPU/CUDA type mismatch (same as M2 fix).
            _M3_PIPE = _M3_PIPE.to(device)
            try:
                _M3_PIPE.vae.enable_tiling()
            except Exception:
                pass
            _M3_PIPE.set_progress_bar_config(disable=True)
            log.info("M3 Kontext base pipe loaded on %s (class=%s)", device, _KontextCls.__name__)

        if _M3_USE_LORA:
            if _M3_CURRENT_PROCEDURE != procedure:
                from peft import PeftModel
                if not isinstance(_M3_PIPE.transformer, PeftModel):
                    # Wrap with the first adapter
                    _M3_PIPE.transformer = PeftModel.from_pretrained(
                        _M3_PIPE.transformer, str(lora_path), adapter_name=procedure
                    )
                else:
                    # Load subsequent adapter
                    _M3_PIPE.transformer.load_adapter(str(lora_path), adapter_name=procedure)
                
                _M3_PIPE.transformer.set_adapter(procedure)
                _M3_PIPE.transformer.eval()
                _M3_CURRENT_PROCEDURE = procedure
                log.info("M3 Kontext PEFT adapter loaded and set to %s (%s)", procedure, lora_path)
        else:
            # Zero-shot mode: if it's a PeftModel, disable all adapters.
            from peft import PeftModel
            if isinstance(_M3_PIPE.transformer, PeftModel):
                _M3_PIPE.transformer.base_model.disable_adapter_layers()
            _M3_CURRENT_PROCEDURE = None
            log.info("M3 Kontext zero-shot mode active for %s (no LoRA)", procedure)

        return _M3_PIPE
    except Exception as e:
        log.warning("M3 Kontext load failed (%s); method will stay disabled", e)
        _M3_LOAD_FAILED = True
        return None


def _gen_m3_kontext(ctx: CandidateContext, seed: int) -> np.ndarray:
    """M3: FLUX.1-Kontext + paired LoRA inference.

    Kontext is a text-conditioned image-to-image model; no mask is passed.
    The LoRA was trained on HDA pre-to-post pairs per procedure, so the
    adapter is procedure-specific.
    """
    import torch
    from PIL import Image

    pipe = _load_m3_kontext_pipe(ctx.procedure)
    if pipe is None:
        raise RuntimeError(
            f"M3 Kontext pipeline unavailable for {ctx.procedure}; "
            "ensure checkpoints/kontext_standalone/<procedure>/final exists"
        )

    h, w = ctx.input_bgr.shape[:2]
    target = 1024

    pil_img = ctx.warped_pil
    if pil_img is None:
        src = ctx.warped_bgr if ctx.warped_bgr is not None else ctx.input_bgr
        pil_img = Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
    pil_img = pil_img.resize((target, target), Image.LANCZOS)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    result = pipe(
        prompt=ctx.preset_prompt,
        image=pil_img,
        height=target,
        width=target,
        guidance_scale=3.5,
        num_inference_steps=20,
        generator=gen,
    )
    out_pil = result.images[0].resize((w, h), Image.LANCZOS)
    return cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)


def enable_m3_kontext(procedure: str | None = None,
                       use_lora: bool = False) -> bool:
    """Attempt to load Kontext in zero-shot or LoRA mode; flip M3 on if successful.

    If procedure is None, tries rhinoplasty as the probe (most common).
    Returns True on successful load. Unlabeled callers default to
    zero-shot because the trained Kontext LoRAs on disk (checkpoints/
    kontext_standalone/*) produce passthrough per iter #13 audit, whereas
    zero-shot FLUX.1-Kontext-dev produces visibly modified outputs
    (iter #39 probe, job 10181103: bleph arc(out,gt)=0.603 /
    outside_ssim=0.963; rhino 0.427 / 0.994).

    The global _M3_USE_LORA flag is restored if the underlying pipeline
    loader fails, so a failed call does not silently poison subsequent
    calls with a different use_lora argument.
    """
    global _M3_USE_LORA
    prior_flag = _M3_USE_LORA
    _M3_USE_LORA = bool(use_lora)
    probe = procedure or "rhinoplasty"
    try:
        pipe = _load_m3_kontext_pipe(probe)
    except Exception:
        _M3_USE_LORA = prior_flag
        raise
    if pipe is None:
        _M3_USE_LORA = prior_flag
        return False
    meta = METHODS["M3_kontext"]
    mode_suffix = " (LoRA)" if _M3_USE_LORA else " (zero-shot)"
    base_label = meta.label.split(" (")[0]
    new_label = base_label + mode_suffix
    METHODS["M3_kontext"] = Method(
        **{**meta.__dict__, "enabled": True, "label": new_label},
    )
    log.info("Method M3_kontext enabled (probed with %s, use_lora=%s, label=%r)",
             probe, _M3_USE_LORA, new_label)
    return True


# ---------------------------------------------------------------------------
# M4: FLUX.1-Fill-dev + mask-aware LoRA (trained on HDA)
# ---------------------------------------------------------------------------

FILL_LORA_PATHS: dict[str, str] = {
    "rhinoplasty": "checkpoints/filldev/rhinoplasty/final",
    "blepharoplasty": "checkpoints/filldev/blepharoplasty/final",
    "rhytidectomy": "checkpoints/filldev/rhytidectomy/final",
}

_M4_PIPE = None
_M4_CURRENT_PROCEDURE: str | None = None
_M4_LOAD_FAILED = False


def _load_m4_fill_lora_pipe(procedure: str) -> Any | None:
    """Load FLUX.1-Fill-dev + procedure-specific PEFT mask-aware LoRA.

    Same PEFT-format pattern as M3: the checkpoint has keys like
    `base_model.model.double_blocks.0.img_mod.lin.lora_A.weight`, which
    are incompatible with diffusers' `load_lora_weights()`. We use
    `PeftModel.from_pretrained` on `pipe.transformer` to apply them
    correctly, matching `scripts/run_production.load_flux_pipeline`.
    """
    global _M4_PIPE, _M4_CURRENT_PROCEDURE, _M4_LOAD_FAILED

    if _M4_LOAD_FAILED:
        return None

    lora_rel = FILL_LORA_PATHS.get(procedure)
    if lora_rel is None:
        log.warning("M4: no Fill LoRA registered for procedure %r", procedure)
        return None
    lora_path = _resolve_checkpoint_path(lora_rel)
    if lora_path is None:
        log.warning("M4: Fill LoRA checkpoint not found at %s", lora_rel)
        return None

    try:
        import os
        import torch
        from diffusers import FluxFillPipeline
        from peft import PeftModel

        if _M4_PIPE is None:
            token = os.environ.get("HF_TOKEN")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            _M4_PIPE = FluxFillPipeline.from_pretrained(
                ICEDIT_BASE_MODEL, torch_dtype=dtype, token=token,
            )
            _M4_PIPE = _M4_PIPE.to(device)
            try:
                _M4_PIPE.vae.enable_tiling()
            except Exception:
                pass
            _M4_PIPE.set_progress_bar_config(disable=True)
            log.info("M4 Fill base pipe loaded on %s", device)

        if _M4_CURRENT_PROCEDURE != procedure:
            from peft import PeftModel
            if not isinstance(_M4_PIPE.transformer, PeftModel):
                _M4_PIPE.transformer = PeftModel.from_pretrained(
                    _M4_PIPE.transformer, str(lora_path), adapter_name=procedure
                )
            else:
                _M4_PIPE.transformer.load_adapter(str(lora_path), adapter_name=procedure)
            
            _M4_PIPE.transformer.set_adapter(procedure)
            _M4_PIPE.transformer.eval()
            _M4_CURRENT_PROCEDURE = procedure
            log.info("M4 Fill PEFT adapter loaded and set to %s (%s)", procedure, lora_path)

        return _M4_PIPE
    except Exception as e:
        log.warning("M4 Fill LoRA load failed (%s); method will stay disabled", e)
        _M4_LOAD_FAILED = True
        return None


def _gen_m4_fill_lora(ctx: CandidateContext, seed: int) -> np.ndarray:
    """M4: FLUX.1-Fill-dev + mask-aware LoRA inference.

    Same input surface as M2 (image + mask + prompt). The difference is
    the LoRA: M2 uses the pretrained ICEdit MoE-LoRA, M4 uses our HDA-
    trained mask-aware adapter.
    """
    import torch
    from PIL import Image

    pipe = _load_m4_fill_lora_pipe(ctx.procedure)
    if pipe is None:
        raise RuntimeError(
            f"M4 Fill-LoRA pipeline unavailable for {ctx.procedure}; "
            "ensure checkpoints/filldev/<procedure>/final exists"
        )

    h, w = ctx.input_bgr.shape[:2]
    target = 1024
    size = (target, target)

    pil_img = ctx.warped_pil
    if pil_img is None:
        src = ctx.warped_bgr if ctx.warped_bgr is not None else ctx.input_bgr
        pil_img = Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
    pil_img = pil_img.resize(size, Image.LANCZOS)

    mask = ctx.mask
    if mask.shape[:2] != size:
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR)
    if mask.dtype == np.float32 or mask.dtype == np.float64:
        mask_u8 = np.clip(mask * 255, 0, 255).astype(np.uint8)
    else:
        mask_u8 = mask.astype(np.uint8)
    mask_pil = Image.fromarray(mask_u8)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    result = pipe(
        prompt=ctx.preset_prompt,
        image=pil_img,
        mask_image=mask_pil,
        height=target,
        width=target,
        strength=ctx.inpainting_strength,
        guidance_scale=3.5,
        num_inference_steps=20,
        generator=gen,
    )
    out_pil = result.images[0].resize((w, h), Image.LANCZOS)
    return cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)


def enable_m4_fill_lora(procedure: str | None = None) -> bool:
    """Attempt to load Fill + LoRA for one procedure; flip M4 on if successful."""
    probe = procedure or "rhinoplasty"
    pipe = _load_m4_fill_lora_pipe(probe)
    if pipe is None:
        return False
    meta = METHODS["M4_fill_lora"]
    METHODS["M4_fill_lora"] = Method(**{**meta.__dict__, "enabled": True})
    log.info("Method M4_fill_lora enabled (probed with %s)", probe)
    return True


def enable_all_trained_methods(procedure: str | None = None,
                                m3_use_lora: bool = False) -> dict[str, bool]:
    """Try to load every trained diffusion method. Returns {method_key: enabled}.

    Call once at startup on a GPU node. Any method whose weights load
    successfully is enabled in the METHODS registry; any that fails
    (missing checkpoint, no HF_TOKEN, OOM) stays disabled and logs a
    warning. M1 and M5 are always enabled by default.

    m3_use_lora defaults to False because the trained Kontext LoRAs on
    disk are passthrough; zero-shot FLUX.1-Kontext-dev produces visibly
    modified outputs.
    """
    results = {
        "M2_icedit": enable_m2_icedit(),
        "M3_kontext": enable_m3_kontext(procedure, use_lora=m3_use_lora),
        "M4_fill_lora": enable_m4_fill_lora(procedure),
    }
    log.info("enable_all_trained_methods: %s (m3_use_lora=%s)",
             results, m3_use_lora)
    return results


def _gen_stub(method_key: str) -> Callable[[CandidateContext, int], np.ndarray]:
    def _stub(ctx: CandidateContext, seed: int) -> np.ndarray:
        raise NotImplementedError(
            f"Method {method_key} not yet implemented (training pending)."
        )

    return _stub


METHOD_IMPL: dict[str, Callable[[CandidateContext, int], np.ndarray]] = {
    "M1_flux_zs": _gen_m1_flux_zs,
    "M2_icedit": _gen_m2_icedit,
    "M3_kontext": _gen_m3_kontext,
    "M4_fill_lora": _gen_m4_fill_lora,
    TPS_METHOD: _gen_m5_tps,
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_candidates(
    ctx: CandidateContext,
    *,
    seeds: list[int] | None = None,
    methods: list[str] | None = None,
) -> list[Candidate]:
    """Generate all candidates (enabled methods x seeds) for one case.

    Always emits at least one TPS fallback candidate (contract with
    `scorer.select_best`). Disabled or stubbed methods are skipped
    with a log line; the pipeline does not crash if one method fails.

    Args:
        ctx: Shared context produced by `prepare_context`.
        seeds: Seed list. Defaults to DEFAULT_SEEDS.
        methods: Method keys to run. Defaults to all currently-enabled
                 methods in METHODS.

    Returns:
        List of Candidate objects. Guaranteed to include one TPS entry.
    """
    seeds = list(seeds) if seeds else list(DEFAULT_SEEDS)
    methods = list(methods) if methods else [k for k, m in METHODS.items() if m.enabled]

    # Enforce TPS presence
    if TPS_METHOD not in methods:
        methods.append(TPS_METHOD)

    out: list[Candidate] = []
    for method_key in methods:
        meta = METHODS.get(method_key)
        if meta is None:
            log.warning("Unknown method %r; skipping", method_key)
            continue
        if not meta.enabled:
            log.info("Method %s disabled; skipping", method_key)
            continue

        impl = METHOD_IMPL[method_key]
        method_seeds: list[int] = [seeds[0]] if method_key == TPS_METHOD else seeds

        for seed in method_seeds:
            try:
                img = impl(ctx, seed)
            except NotImplementedError:
                log.info("Method %s not implemented yet; skipping", method_key)
                break
            except Exception as e:  # pragma: no cover - infrastructure failure
                log.error("Method %s seed=%d failed: %s", method_key, seed, e)
                continue

            out.append(
                Candidate(
                    image_bgr=img,
                    method=method_key,
                    seed=seed,
                    metadata={"label": meta.label, "procedure": ctx.procedure},
                )
            )

    if not any(c.is_fallback for c in out):
        log.error(
            "generate_candidates produced no TPS fallback -- pipeline contract violated. "
            "This means _gen_m5_tps raised before returning."
        )

    n_diff = sum(1 for c in out if not c.is_fallback)
    n_tps = sum(1 for c in out if c.is_fallback)
    log.info(
        "generate_candidates: %d diffusion + %d TPS = %d total (methods=%s, seeds=%s)",
        n_diff, n_tps, len(out), methods, seeds,
    )
    return out


def enabled_methods() -> list[str]:
    """List currently-enabled method keys."""
    return [k for k, m in METHODS.items() if m.enabled]


def enable_method(key: str) -> None:
    """Enable a method at runtime (used after training completes)."""
    if key not in METHODS:
        raise KeyError(f"Unknown method {key!r}")
    METHODS[key] = Method(**{**METHODS[key].__dict__, "enabled": True})
    log.info("Method %s enabled", key)


def disable_method(key: str) -> None:
    if key not in METHODS:
        raise KeyError(f"Unknown method {key!r}")
    METHODS[key] = Method(**{**METHODS[key].__dict__, "enabled": False})
    log.info("Method %s disabled", key)
