"""MediaPipe 478-landmark face extraction module.

Extracts dense facial landmarks for:
  - Surgical mask generation (nose, eyelid, jaw regions)
  - Depth map alignment
  - TPS warp keypoint selection
  - Identity-aware region segmentation

Supports both the legacy mp.solutions API and the newer tasks API
(mediapipe >= 0.10.14 / Python 3.13+).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Landmark region indices (MediaPipe 478-point mesh)
# ---------------------------------------------------------------------------
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

NOSE_DORSUM = [6, 168, 197, 195, 5, 4]
NOSE_TIP = [1, 2, 3, 4, 5, 6, 19, 94, 164, 168, 195, 197]
NOSE_WINGS = [
    45, 51, 122, 188, 114, 217, 126, 142, 97,
    275, 281, 351, 412, 343, 437, 355, 371, 326,
]
NOSE_ALL = sorted(set(NOSE_DORSUM + NOSE_TIP + NOSE_WINGS))

LEFT_EYE_UPPER = [246, 161, 160, 159, 158, 157, 173]
LEFT_EYE_LOWER = [33, 7, 163, 144, 145, 153, 154, 155, 133]
RIGHT_EYE_UPPER = [466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_LOWER = [263, 249, 390, 373, 374, 380, 381, 382, 362]

LEFT_EYELID = LEFT_EYE_UPPER + LEFT_EYE_LOWER
RIGHT_EYELID = RIGHT_EYE_UPPER + RIGHT_EYE_LOWER
EYELIDS_ALL = sorted(set(LEFT_EYELID + RIGHT_EYELID))

# Upper eyelid skin fold + brow region (for blepharoplasty mask).
# Extends the convex hull from the eyebrow down to the lash line so the
# hard-mask composite allows the model to edit the full hooded skin region.
# Lid margin: 246/466-series (upper lash line)
# Periorbital fold: 56, 28-30 (medial) / 286, 258-260 (right medial)
# Left brow (FACEMESH_LEFT_EYEBROW): 55, 65, 52, 53, 46, 70, 63, 105, 66, 107
# Right brow (FACEMESH_RIGHT_EYEBROW): 285, 295, 282, 283, 276, 300, 293, 334, 296, 336
LEFT_UPPER_LID_FOLD = [
    246, 161, 160, 159, 158, 157, 173,  # upper lash line
    56, 28, 27, 29, 30,                  # periorbital fold
    55, 65, 52, 53, 46, 70, 63, 105, 66, 107,  # left eyebrow
]
RIGHT_UPPER_LID_FOLD = [
    466, 388, 387, 386, 385, 384, 398,   # upper lash line
    286, 258, 257, 259, 260,             # periorbital fold
    285, 295, 282, 283, 276, 300, 293, 334, 296, 336,  # right eyebrow
]
UPPER_LIDS_ONLY = sorted(set(LEFT_UPPER_LID_FOLD + RIGHT_UPPER_LID_FOLD))

JAW_CONTOUR = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

# Lower jaw only: ear → mandible → chin → ear. NO temples/forehead/cheeks.
LOWER_JAW = [
    # Left: ear to chin
    454, 323, 361, 288, 397, 365,
    # Chin: left to right
    379, 378, 400, 377, 152, 148, 176, 149, 150,
    # Right: chin to ear
    136, 172, 58, 132,
]
CHIN = [152, 148, 176, 149, 150, 136, 172, 58, 132, 377, 400, 378, 379, 365, 397]

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

# Surgical procedure -> landmark indices
PROCEDURE_LANDMARKS: dict[str, list[int]] = {
    "rhinoplasty": NOSE_ALL,
    "blepharoplasty": UPPER_LIDS_ONLY,
    "orthognathic": JAW_CONTOUR + CHIN,
    "rhytidectomy": JAW_CONTOUR,  # jaw contour only; generate_adaptive_rhytid_mask extends to neck
}


class FaceLandmarks(NamedTuple):
    """Result of landmark extraction."""

    points: np.ndarray  # (478, 2) float32, pixel coordinates
    confidence: float  # detection confidence
    image_size: tuple[int, int]  # (width, height)


def extract_landmarks(
    image: np.ndarray | str | Path,
) -> FaceLandmarks | None:
    """Extract 478 face landmarks from an image.

    Args:
        image: BGR numpy array, or path to image file.

    Returns:
        FaceLandmarks or None if no face detected.
    """
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
        if image is None:
            log.error("Could not read image: %s", image)
            return None

    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Try legacy API first
    result = _extract_legacy(rgb, w, h)
    if result is not None:
        return result

    # Fall back to tasks API
    result = _extract_tasks(rgb, w, h)
    if result is not None:
        return result

    log.warning("No face detected by any MediaPipe backend")
    return None


def _extract_legacy(rgb: np.ndarray, w: int, h: int) -> FaceLandmarks | None:
    """Extract landmarks using mp.solutions (mediapipe < 0.10.14)."""
    try:
        import mediapipe as mp

        face_mesh = mp.solutions.face_mesh
    except (ImportError, AttributeError):
        return None

    with face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as mesh:
        results = mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None

        lms = results.multi_face_landmarks[0]
        points = np.array(
            [(lm.x * w, lm.y * h) for lm in lms.landmark],
            dtype=np.float32,
        )
        log.info("Extracted %d landmarks (legacy API)", len(points))
        return FaceLandmarks(points=points, confidence=1.0, image_size=(w, h))


def _extract_tasks(rgb: np.ndarray, w: int, h: int) -> FaceLandmarks | None:
    """Extract landmarks using MediaPipe tasks API (>= 0.10.14)."""
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError:
        return None

    import urllib.request

    model_path = Path("/tmp/envisage_face_landmarker.task")
    if not model_path.exists():
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )
        log.info("Downloading MediaPipe face landmarker model...")
        urllib.request.urlretrieve(url, str(model_path))

    try:
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            num_faces=1,
        )
        detector = mp_vision.FaceLandmarker.create_from_options(options)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        if not result.face_landmarks:
            return None

        face_lms = result.face_landmarks[0]
        points = np.array(
            [(lm.x * w, lm.y * h) for lm in face_lms],
            dtype=np.float32,
        )
        log.info("Extracted %d landmarks (tasks API)", len(points))
        return FaceLandmarks(points=points, confidence=1.0, image_size=(w, h))
    except Exception as e:
        log.warning("Tasks API failed: %s", e)
        return None


def get_region_points(
    landmarks: FaceLandmarks,
    procedure: str,
) -> np.ndarray:
    """Get landmark points for a specific surgical procedure region.

    Args:
        landmarks: Full 478-point landmarks.
        procedure: One of 'rhinoplasty', 'blepharoplasty', 'rhytidectomy', 'orthognathic'.

    Returns:
        (N, 2) array of landmark points for the procedure region.
    """
    indices = PROCEDURE_LANDMARKS.get(procedure)
    if indices is None:
        raise ValueError(
            f"Unknown procedure: {procedure}. "
            f"Choose from: {list(PROCEDURE_LANDMARKS.keys())}"
        )
    valid = [i for i in indices if i < len(landmarks.points)]
    return landmarks.points[valid]


# ---------------------------------------------------------------------------
# Anatomical measurements from landmarks
# ---------------------------------------------------------------------------

def measure_nose(landmarks: FaceLandmarks) -> dict[str, float]:
    """Measure nose dimensions from landmarks.

    Returns:
        Dict with 'width' (px), 'height' (px), 'center_x', 'center_y',
        'left_alar_x', 'right_alar_x'.
    """
    pts = landmarks.points
    # Alar width: distance between landmarks 48 (left) and 278 (right)
    left_alar = pts[48] if 48 < len(pts) else pts[0]
    right_alar = pts[278] if 278 < len(pts) else pts[0]
    width = float(np.linalg.norm(left_alar - right_alar))

    # Nose height: nasion (landmark 6) to subnasale/tip (landmark 1)
    nasion = pts[6] if 6 < len(pts) else pts[0]
    tip = pts[1] if 1 < len(pts) else pts[0]
    height = float(np.linalg.norm(nasion - tip))

    center_x = float((left_alar[0] + right_alar[0]) / 2)
    center_y = float((nasion[1] + tip[1]) / 2)

    return {
        "width": width,
        "height": height,
        "center_x": center_x,
        "center_y": center_y,
        "left_alar_x": float(left_alar[0]),
        "right_alar_x": float(right_alar[0]),
        "nasion_y": float(nasion[1]),
        "tip_y": float(tip[1]),
    }


def measure_nasal_symmetry(landmarks: FaceLandmarks) -> dict[str, float]:
    """Measure nasal symmetry metrics for personalized prompting.

    Returns:
        Dict with alar_symmetry_ratio (1.0=perfect), tip_deviation_px,
        dorsal_deviation_std, bridge_width_ratio (vs intercanthal distance).
    """
    pts = landmarks.points

    # Facial midline from nasion through philtrum
    nasion = pts[6] if 6 < len(pts) else pts[0]
    philtrum = pts[164] if 164 < len(pts) else pts[0]
    midline_x = (nasion[0] + philtrum[0]) / 2.0

    # Alar symmetry
    left_alar = pts[48] if 48 < len(pts) else pts[0]
    right_alar = pts[278] if 278 < len(pts) else pts[0]
    left_dist = abs(midline_x - left_alar[0])
    right_dist = abs(right_alar[0] - midline_x)
    alar_symmetry = min(left_dist, right_dist) / max(left_dist, right_dist, 1.0)

    # Tip deviation from midline
    tip = pts[1] if 1 < len(pts) else pts[0]
    tip_deviation = float(tip[0] - midline_x)

    # Dorsal midline deviation (straightness)
    bridge_pts = pts[[i for i in NOSE_DORSUM if i < len(pts)]]
    dorsal_deviation = float(np.std(bridge_pts[:, 0] - midline_x)) if len(bridge_pts) > 2 else 0.0

    # Alar width vs intercanthal distance
    left_inner = pts[133] if 133 < len(pts) else pts[33]
    right_inner = pts[362] if 362 < len(pts) else pts[263]
    icd = abs(right_inner[0] - left_inner[0])
    alar_width = abs(right_alar[0] - left_alar[0])
    bridge_width_ratio = alar_width / max(icd, 1.0)

    # Tip width (bulbosity); use actual tip landmarks, not alar base
    # Landmarks 2 (left tip), 98 (right tip) define the tip lobule width
    left_tip_lm = pts[2] if 2 < len(pts) else pts[1]
    right_tip_lm = pts[98] if 98 < len(pts) else pts[1]
    tip_width = abs(right_tip_lm[0] - left_tip_lm[0])
    tip_bulbosity = tip_width / max(alar_width, 1.0)

    return {
        "alar_symmetry_ratio": float(alar_symmetry),
        "tip_deviation_px": tip_deviation,
        "dorsal_deviation_std": dorsal_deviation,
        "bridge_width_ratio": float(bridge_width_ratio),
        "tip_bulbosity": float(tip_bulbosity),
        "midline_x": float(midline_x),
        "alar_width": float(alar_width),
        "intercanthal_distance": float(icd),
    }


def measure_eyelid_hooding(landmarks: FaceLandmarks) -> dict[str, float]:
    """Measure eyelid hooding for each eye independently.

    Hooding is estimated as the distance from the upper lid crease to the
    brow minus the distance from the crease to the lash line. A smaller
    value indicates more hooding (skin covers more of the lid fold).

    Returns:
        Dict with 'left_hooding', 'right_hooding', 'left_crease_to_brow',
        'right_crease_to_brow', 'asymmetry'.
    """
    pts = landmarks.points

    # Left eye: upper lid center (159), brow center (105), lower lid (145)
    left_lid = pts[159] if 159 < len(pts) else pts[0]
    left_brow = pts[105] if 105 < len(pts) else pts[0]
    left_lower = pts[145] if 145 < len(pts) else pts[0]

    # Right eye: upper lid center (386), brow center (334), lower lid (374)
    right_lid = pts[386] if 386 < len(pts) else pts[0]
    right_brow = pts[334] if 334 < len(pts) else pts[0]
    right_lower = pts[374] if 374 < len(pts) else pts[0]

    left_crease_to_brow = float(abs(left_brow[1] - left_lid[1]))
    left_crease_to_lash = float(abs(left_lid[1] - left_lower[1]))
    right_crease_to_brow = float(abs(right_brow[1] - right_lid[1]))
    right_crease_to_lash = float(abs(right_lid[1] - right_lower[1]))

    # Hooding: less visible lid fold = more hooding
    # Ratio of crease-to-brow / crease-to-lash: lower = more hooded
    left_hooding = left_crease_to_brow / max(left_crease_to_lash, 1.0)
    right_hooding = right_crease_to_brow / max(right_crease_to_lash, 1.0)

    asymmetry = abs(left_hooding - right_hooding)

    return {
        "left_hooding": left_hooding,
        "right_hooding": right_hooding,
        "left_crease_to_brow": left_crease_to_brow,
        "right_crease_to_brow": right_crease_to_brow,
        "asymmetry": asymmetry,
    }


def measure_jaw(landmarks: FaceLandmarks) -> dict[str, float]:
    """Measure jaw contour dimensions.

    Returns:
        Dict with 'jaw_width', 'jaw_mean_y', 'chin_y', and
        'jaw_contour_points' (list of (x, y) tuples along the jaw).
    """
    pts = landmarks.points
    jaw_pts = pts[[i for i in JAW_CONTOUR if i < len(pts)]]

    if len(jaw_pts) < 3:
        w, h = landmarks.image_size
        return {
            "jaw_width": w * 0.6,
            "jaw_mean_y": h * 0.7,
            "chin_y": h * 0.85,
            "jaw_contour_points": [],
        }

    jaw_width = float(jaw_pts[:, 0].max() - jaw_pts[:, 0].min())
    jaw_mean_y = float(jaw_pts[:, 1].mean())
    chin_y = float(pts[152][1]) if 152 < len(pts) else float(jaw_pts[:, 1].max())

    return {
        "jaw_width": jaw_width,
        "jaw_mean_y": jaw_mean_y,
        "chin_y": chin_y,
        "jaw_contour_points": jaw_pts.tolist(),
    }


def estimate_head_pose(landmarks: FaceLandmarks) -> dict[str, float]:
    """Estimate head yaw from landmark asymmetry.

    Returns:
        Dict with 'yaw_degrees' (approximate).
    """
    pts = landmarks.points
    w = landmarks.image_size[0]

    # Use eye outer corners as reference
    left_eye = pts[33] if 33 < len(pts) else np.array([w * 0.35, 0])
    right_eye = pts[263] if 263 < len(pts) else np.array([w * 0.65, 0])
    nose_tip = pts[1] if 1 < len(pts) else np.array([w * 0.5, 0])

    # If nose tip is centered between eyes, yaw ~ 0
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    eye_span = abs(right_eye[0] - left_eye[0])
    if eye_span < 1:
        return {"yaw_degrees": 0.0}

    # Ratio of nose offset from eye center to eye span
    nose_offset = (nose_tip[0] - eye_center_x) / eye_span
    yaw = float(np.degrees(np.arcsin(np.clip(nose_offset * 2, -1, 1))))

    return {"yaw_degrees": yaw}


def draw_landmarks(
    image: np.ndarray,
    landmarks: FaceLandmarks,
    procedure: str | None = None,
    radius: int = 1,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw landmarks on an image copy.

    Args:
        image: BGR image.
        landmarks: Extracted landmarks.
        procedure: If given, highlight only that procedure's region.
        radius: Point radius.
        color: BGR color for points.

    Returns:
        Annotated image copy.
    """
    vis = image.copy()
    if procedure is not None:
        pts = get_region_points(landmarks, procedure)
    else:
        pts = landmarks.points

    for x, y in pts.astype(np.int32):
        cv2.circle(vis, (x, y), radius, color, -1)
    return vis
