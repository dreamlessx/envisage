"""envisage:Facial Surgery Outcome Prediction Demo.

Gradio app for predicting rhinoplasty outcomes using FLUX.1-dev
inpainting with depth-conditioned ControlNet.

Deploy: gradio app.py
HF Spaces: set app_file=app.py in README.md

Requires: GPU with >= 24GB VRAM (A10G, L40S, A100, or A6000)
"""

from __future__ import annotations

import gc
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from envisage.landmarks import (
    FaceLandmarks,
    NOSE_DORSUM,
    NOSE_TIP,
    NOSE_WINGS,
    extract_landmarks,
)
from envisage.masks import MaskConfig, generate_mask, mask_to_pil
from envisage.depth import DepthEstimator, DepthModConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


# ---------------------------------------------------------------------------
# Rhinoplasty sub-type depth modifications
# ---------------------------------------------------------------------------

@dataclass
class RhinoplastySubType:
    """Parameters for a rhinoplasty sub-procedure."""

    name: str
    description: str
    center_landmark: int
    sigma_x_frac: float
    sigma_y_frac: float
    base_intensity: float  # scaled by user's intensity slider


RHINOPLASTY_SUBTYPES = {
    "Dorsal Hump Reduction": RhinoplastySubType(
        name="dorsal_hump",
        description="Flatten the nasal bridge by reducing the dorsal hump",
        center_landmark=6,  # nasion (bridge)
        sigma_x_frac=0.05,
        sigma_y_frac=0.10,
        base_intensity=50.0,
    ),
    "Tip Refinement": RhinoplastySubType(
        name="tip_refine",
        description="Refine the nasal tip shape and projection",
        center_landmark=4,  # tip of nose
        sigma_x_frac=0.06,
        sigma_y_frac=0.05,
        base_intensity=35.0,
    ),
    "Alar Narrowing": RhinoplastySubType(
        name="alar_narrow",
        description="Narrow the nasal wings (alae)",
        center_landmark=94,  # between wings
        sigma_x_frac=0.10,
        sigma_y_frac=0.04,
        base_intensity=30.0,
    ),
}


def modify_depth_subtype(
    depth: np.ndarray,
    landmarks: FaceLandmarks,
    mask: np.ndarray,
    subtype: RhinoplastySubType,
    intensity_pct: float,
) -> np.ndarray:
    """Modify depth map for a specific rhinoplasty sub-type.

    Args:
        depth: (H, W) float32 depth map.
        landmarks: Face landmarks.
        mask: (H, W) float32 mask [0, 1].
        subtype: Sub-procedure parameters.
        intensity_pct: User intensity 0-100%.
    """
    h, w = depth.shape[:2]
    modified = depth.copy()

    intensity = subtype.base_intensity * (intensity_pct / 100.0)

    if subtype.center_landmark < len(landmarks.points):
        cx = int(landmarks.points[subtype.center_landmark][0])
        cy = int(landmarks.points[subtype.center_landmark][1])
    else:
        cx, cy = w // 2, int(h * 0.50)

    sigma_x = w * subtype.sigma_x_frac
    sigma_y = h * subtype.sigma_y_frac

    y_coords, x_coords = np.mgrid[0:h, 0:w]
    gaussian = np.exp(-(
        (x_coords - cx) ** 2 / (2 * sigma_x ** 2) +
        (y_coords - cy) ** 2 / (2 * sigma_y ** 2)
    ))

    modified -= gaussian * intensity

    # Mask-aware blending
    mask_r = mask
    if mask_r.shape[:2] != (h, w):
        mask_r = cv2.resize(mask_r, (w, h))
    if mask_r.max() > 1.0:
        mask_r = mask_r / 255.0

    modified = mask_r * modified + (1 - mask_r) * depth
    return np.clip(modified, 0, 255)


# ---------------------------------------------------------------------------
# Pipeline singleton (lazy-loaded)
# ---------------------------------------------------------------------------

_pipeline_cache: dict = {}


def get_pipeline():
    """Load FLUX + ControlNet pipeline (cached)."""
    if "pipe" in _pipeline_cache:
        return _pipeline_cache["pipe"]

    token = os.environ.get("HF_TOKEN")

    log.info("Loading FLUX pipeline with ControlNet...")

    # Try ControlNet inpainting pipeline first, then fallback
    try:
        from diffusers import FluxControlNetInpaintPipeline, FluxControlNetModel

        controlnet = FluxControlNetModel.from_pretrained(
            "jasperai/Flux.1-dev-Controlnet-Depth",
            torch_dtype=DTYPE, token=token,
        )
        pipe = FluxControlNetInpaintPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            controlnet=controlnet,
            torch_dtype=DTYPE, token=token,
        )
        pipe.enable_model_cpu_offload()
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass
        _pipeline_cache["pipe"] = pipe
        _pipeline_cache["has_controlnet"] = True
        log.info("Loaded FluxControlNetInpaintPipeline")
        return pipe

    except Exception as e:
        log.warning("ControlNet inpainting pipeline failed: %s", e)

    # Fallback: plain inpainting
    from diffusers import FluxInpaintPipeline

    pipe = FluxInpaintPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=DTYPE, token=token,
    )
    pipe.enable_model_cpu_offload()
    _pipeline_cache["pipe"] = pipe
    _pipeline_cache["has_controlnet"] = False
    log.info("Loaded FluxInpaintPipeline (no ControlNet)")
    return pipe


def get_depth_estimator():
    """Get or create depth estimator (cached)."""
    if "depth" not in _pipeline_cache:
        _pipeline_cache["depth"] = DepthEstimator(
            device=0 if DEVICE == "cuda" else -1
        )
    return _pipeline_cache["depth"]


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------

def predict(
    input_image: Image.Image,
    subtype_name: str,
    intensity: float,
    num_steps: int,
) -> tuple[Image.Image, Image.Image, Image.Image, str]:
    """Run rhinoplasty prediction.

    Returns:
        (output_image, mask_image, depth_comparison, status_text)
    """
    if input_image is None:
        return None, None, None, "Please upload a face image."

    subtype = RHINOPLASTY_SUBTYPES.get(subtype_name)
    if subtype is None:
        return None, None, None, f"Unknown sub-type: {subtype_name}"

    try:
        # Convert to BGR for OpenCV
        img_bgr = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)

        # Extract landmarks
        landmarks = extract_landmarks(img_bgr)
        if landmarks is None:
            return None, None, None, "No face detected. Please upload a clear frontal face photo."

        # Generate mask
        mask = generate_mask(landmarks, "rhinoplasty", MaskConfig(dilation_px=25, feather_sigma=15))

        # Depth estimation
        estimator = get_depth_estimator()
        depth = estimator.estimate(input_image)

        # Modify depth for sub-type
        modified_depth = modify_depth_subtype(depth, landmarks, mask, subtype, intensity)

        # Create depth comparison image
        h, w = depth.shape[:2]
        depth_vis = np.hstack([
            depth.astype(np.uint8),
            modified_depth.astype(np.uint8),
        ])
        depth_comparison = Image.fromarray(depth_vis, mode="L").convert("RGB")

        # Prepare for generation
        target_size = 512
        size = (target_size, target_size)
        image_resized = input_image.resize(size, Image.LANCZOS)

        mask_resized = cv2.resize(mask, size)
        mask_pil = Image.fromarray((mask_resized * 255).astype(np.uint8))

        # Depth conditioning image (RGB)
        depth_mod_resized = cv2.resize(modified_depth, size)
        depth_rgb = np.stack([depth_mod_resized.astype(np.uint8)] * 3, axis=-1)
        control_image = Image.fromarray(depth_rgb)

        prompt = (
            "a photorealistic frontal portrait of the same person, "
            f"natural skin texture, {subtype.description.lower()}, "
            "studio lighting, high quality, detailed"
        )

        pipe = get_pipeline()

        gen_kwargs = {
            "prompt": prompt,
            "image": image_resized,
            "mask_image": mask_pil,
            "height": target_size,
            "width": target_size,
            "strength": 0.65 + 0.20 * (intensity / 100.0),
            "guidance_scale": 3.5,
            "num_inference_steps": num_steps,
            "generator": torch.Generator(device="cpu").manual_seed(42),
        }

        if _pipeline_cache.get("has_controlnet"):
            gen_kwargs["control_image"] = control_image
            gen_kwargs["controlnet_conditioning_scale"] = 0.3 + 0.4 * (intensity / 100.0)

        result = pipe(**gen_kwargs)
        output = result.images[0].resize(input_image.size, Image.LANCZOS)

        # ArcFace measurement
        arcface_score = _measure_arcface(input_image, output)

        status = (
            f"Sub-type: {subtype_name}\n"
            f"Intensity: {intensity:.0f}%\n"
            f"Steps: {num_steps}\n"
            f"ArcFace similarity: {arcface_score:.3f}\n"
            f"Identity preserved: {'Yes' if arcface_score > 0.65 else 'No'}"
        )

        mask_display = mask_to_pil(mask).convert("RGB")

        return output, mask_display, depth_comparison, status

    except Exception as e:
        log.error("Prediction failed: %s", e, exc_info=True)
        return None, None, None, f"Error: {e}"


def _measure_arcface(input_img: Image.Image, output_img: Image.Image) -> float:
    """Compute ArcFace similarity."""
    try:
        from insightface.app import FaceAnalysis

        if "arcface" not in _pipeline_cache:
            app = FaceAnalysis(
                name="buffalo_l",
                root=str(Path.home() / ".insightface"),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0 if DEVICE == "cuda" else -1, det_size=(640, 640))
            _pipeline_cache["arcface"] = app

        app = _pipeline_cache["arcface"]
        in_arr = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2BGR)
        out_arr = cv2.cvtColor(np.array(output_img), cv2.COLOR_RGB2BGR)

        in_faces = app.get(in_arr)
        out_faces = app.get(out_arr)
        if not in_faces or not out_faces:
            return float("nan")

        e1, e2 = in_faces[0].embedding, out_faces[0].embedding
        return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

def create_demo() -> gr.Blocks:
    """Build the Gradio demo interface."""
    with gr.Blocks(
        title="envisage:Surgical Outcome Prediction",
    ) as demo:
        gr.Markdown(
            """
            # envisage -- Rhinoplasty Prediction Demo
            Upload a frontal face photo, select a rhinoplasty sub-type, adjust
            intensity, and generate a predicted post-surgical result.

            **How it works:** FLUX.1-dev inpainting with depth-conditioned ControlNet.
            The depth map is modified to simulate the surgical change, then the model
            regenerates the nose region while preserving identity.

            > **Note:** This demo runs rhinoplasty only. The full pipeline (available
            > in the [repository](https://github.com/dreamlessx/envisage_public))
            > also supports blepharoplasty and rhytidectomy with procedure-specific
            > TPS pre-warping, adaptive masks, multi-seed sweep, and two-pass
            > neck/jaw inpainting.

            > **Research use only.** Not a medical device.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Face Photo",
                    type="pil",
                    height=400,
                )
                subtype = gr.Radio(
                    choices=list(RHINOPLASTY_SUBTYPES.keys()),
                    value="Dorsal Hump Reduction",
                    label="Rhinoplasty Sub-Type",
                )
                intensity = gr.Slider(
                    minimum=0, maximum=100, value=50, step=5,
                    label="Intensity (%)",
                    info="How dramatic the surgical change should be",
                )
                num_steps = gr.Slider(
                    minimum=10, maximum=30, value=20, step=5,
                    label="Quality (inference steps)",
                    info="More steps = higher quality but slower",
                )
                generate_btn = gr.Button("Generate Prediction", variant="primary")

            with gr.Column(scale=1):
                output_image = gr.Image(label="Predicted Result", height=400)
                status = gr.Textbox(label="Metrics", lines=5)

        with gr.Row():
            mask_image = gr.Image(label="Surgical Mask", height=200)
            depth_image = gr.Image(label="Depth: Original | Modified", height=200)

        # Subtype descriptions
        with gr.Accordion("Sub-type descriptions", open=False):
            for name, st in RHINOPLASTY_SUBTYPES.items():
                gr.Markdown(f"**{name}**: {st.description}")

        generate_btn.click(
            fn=predict,
            inputs=[input_image, subtype, intensity, num_steps],
            outputs=[output_image, mask_image, depth_image, status],
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )
