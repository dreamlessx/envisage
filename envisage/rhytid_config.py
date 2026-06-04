"""Rhytidectomy sub-procedure taxonomy and auto-detection.

8 frontal-detectable presets. Each preset carries a positive
`prompt_fragment` and an `anchor_fragment`. The anchor clauses are
emitted for every inactive preset so that non-requested facelift
regions are explicitly preserved in the prompt.

Presets:
  1. jawline_straightening   : ruler-straight mandibular border
  2. jowl_elimination        : remove sagging tissue below jawline
  3. neck_smoothing          : smooth taut neck skin (size unchanged)
  4. marionette_softening    : soften marionette lines
  5. platysmal_band_removal  : remove visible vertical neck bands
  6. prejowl_correction      : fill the pre-jowl sulcus
  7. submental_definition    : define cervicomental angle
  8. nasolabial_softening    : soften the lower nasolabial fold only

Mask rules (upstream, masks.py):
- Tight jaw-band around the mandibular contour
- Neck rectangle below chin
- NEVER include eyes, nose, forehead, upper cheeks, mouth
- Preserve neck SIZE (texture-only edit)
- Preserve facial hair
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from .landmarks import FaceLandmarks, measure_jaw

log = logging.getLogger(__name__)


class RhytidZone(str, Enum):
    JAWLINE = "jawline"
    NECK = "neck"
    JOWL = "jowl"


@dataclass(frozen=True)
class RhytidSubProcedure:
    id: int
    key: str
    label: str
    zone: RhytidZone
    prompt_fragment: str
    anchor_fragment: str
    measurement_key: str
    expected_sign: int              # +1/-1/0 (texture-only)
    delta_threshold: float          # baseline atlas-magnitude for MODERATE severity
    detectable_frontal: bool
    description: str


PRIORITY_ORDER = [
    "jawline_straightening",
    "jowl_elimination",
    "neck_smoothing",
    "marionette_softening",
    "platysmal_band_removal",
    "prejowl_correction",
    "submental_definition",
    "nasolabial_softening",
]


RHYTID_PROCEDURES: dict[str, RhytidSubProcedure] = {
    "jawline_straightening": RhytidSubProcedure(
        id=1, key="jawline_straightening",
        label="Jawline Straightening",
        zone=RhytidZone.JAWLINE,
        prompt_fragment="ruler-straight mandibular border from ear to chin like a straight edge",
        anchor_fragment="same mandibular contour as input",
        measurement_key="jaw_sag",
        expected_sign=-1,
        delta_threshold=5.0,
        detectable_frontal=True,
        description="#1 priority. SMAS lift creates a clean jawline with straight-edge contour.",
    ),
    "jowl_elimination": RhytidSubProcedure(
        id=2, key="jowl_elimination",
        label="Jowl Elimination",
        zone=RhytidZone.JOWL,
        prompt_fragment="no jowling or sagging tissue hanging below the mandibular border",
        anchor_fragment="same jowl volume and position as input",
        measurement_key="jaw_sag",
        expected_sign=-1,
        delta_threshold=5.0,
        detectable_frontal=True,
        description="#2 priority. Eliminate tissue ptosis below the jawline.",
    ),
    "neck_smoothing": RhytidSubProcedure(
        id=3, key="neck_smoothing",
        label="Neck Skin Smoothing",
        zone=RhytidZone.NECK,
        prompt_fragment="smooth taut neck skin without wrinkles or texture, identical neck size and proportions",
        anchor_fragment="same neck skin texture as input",
        measurement_key="neck_extent_ratio",
        expected_sign=0,          # neck size must not change; texture-only edit
        delta_threshold=0.02,
        detectable_frontal=True,
        description="Smooth anterior neck without changing size or proportions. Texture-only edit.",
    ),
    "marionette_softening": RhytidSubProcedure(
        id=4, key="marionette_softening",
        label="Marionette Line Softening",
        zone=RhytidZone.JOWL,
        prompt_fragment="softened marionette lines from mouth corners down to jaw",
        anchor_fragment="same marionette lines as input",
        measurement_key="marionette_depth",
        expected_sign=-1,
        delta_threshold=5.0,
        detectable_frontal=True,
        description="SMAS repositioning reduces marionette creases from mouth to jaw.",
    ),
    "platysmal_band_removal": RhytidSubProcedure(
        id=5, key="platysmal_band_removal",
        label="Platysmal Band Removal",
        zone=RhytidZone.NECK,
        prompt_fragment="smooth anterior neck without any visible vertical platysmal bands",
        anchor_fragment="same platysmal band visibility as input",
        measurement_key="neck_extent_ratio",
        expected_sign=0,
        delta_threshold=0.02,
        detectable_frontal=True,
        description="Remove visible vertical neck bands. Smooth texture only.",
    ),
    "prejowl_correction": RhytidSubProcedure(
        id=6, key="prejowl_correction",
        label="Pre-jowl Sulcus Correction",
        zone=RhytidZone.JOWL,
        prompt_fragment="smooth continuous jawline without pre-jowl depression",
        anchor_fragment="same pre-jowl sulcus depth as input",
        measurement_key="jaw_sag",
        expected_sign=-1,
        delta_threshold=5.0,
        detectable_frontal=True,
        description="Fill the depression anterior to the jowl for continuous contour.",
    ),
    "submental_definition": RhytidSubProcedure(
        id=7, key="submental_definition",
        label="Submental Definition",
        zone=RhytidZone.NECK,
        prompt_fragment="clean chin-to-neck transition, same neck width",
        anchor_fragment="same cervicomental angle as input",
        measurement_key="jaw_sag",
        expected_sign=-1,
        delta_threshold=5.0,
        detectable_frontal=True,
        description="Define cervicomental angle while keeping neck size unchanged.",
    ),
    "nasolabial_softening": RhytidSubProcedure(
        id=8, key="nasolabial_softening",
        label="Lower Nasolabial Softening",
        zone=RhytidZone.JOWL,
        prompt_fragment="softened lower nasolabial folds",
        anchor_fragment="same lower nasolabial fold depth as input",
        measurement_key="marionette_depth",
        expected_sign=-1,
        delta_threshold=4.0,
        detectable_frontal=True,
        description="Only the LOWER portion of the nasolabial fold. Upper requires filler.",
    ),
}


class Severity:
    NONE = 0
    MILD = 1
    MODERATE = 2
    SEVERE = 3


SEVERITY_PREFIX: dict[int, str] = {
    Severity.NONE: "",
    Severity.MILD: "slightly ",
    Severity.MODERATE: "",
    Severity.SEVERE: "pronounced ",
}


_BASE_OPENING = "a photorealistic frontal portrait of the same person"

# Rhytid closing scaffold. Priority per clinical feedback: clean jawline,
# smooth neck (texture-only, size unchanged), pixel-identical upper face.
# Wrinkle language is intentionally deprioritized until jawline + neck
# work lands reliably.
_BASE_CLOSING = (
    "ruler-straight smooth continuous jawline edge from ear to chin, "
    "no fat chunks no bulges no doubled jaw contour, "
    "no loose skin patches no irregular skin texture along the jawline, "
    "smooth taut neck skin with identical neck size and proportions, "
    "no change to neck width, no change to neck length, "
    "no dark holes no color drift no weird discoloration, "
    "preserve all facial hair including stubble and beard, "
    "identical mouth shape lip position and chin shape, "
    "upper face pixel-identical to input including eyes nose forehead and cheeks, "
    "sharp focus on jawline and neck edge, crisp skin texture, no blur, no softening, no smoothing artifacts, "
    "no hallucinated anatomy, no extra jaw features, no fabricated structures, "
    "clinical photography lighting, high quality, photorealistic"
)


@dataclass
class RhytidAnalysis:
    detected: dict[str, bool] = field(default_factory=dict)
    severity: dict[str, int] = field(default_factory=dict)
    measurements: dict[str, float] = field(default_factory=dict)

    @property
    def active_procedures(self) -> list[RhytidSubProcedure]:
        return [RHYTID_PROCEDURES[k] for k, v in self.detected.items() if v]

    @property
    def active_keys(self) -> list[str]:
        return [k for k, v in self.detected.items() if v]

    @property
    def inactive_keys(self) -> list[str]:
        active = set(self.active_keys)
        return [k for k in RHYTID_PROCEDURES if k not in active]

    def get_severity(self, key: str) -> int:
        return self.severity.get(key, Severity.MODERATE)

    def build_prompt(self, max_procedures: int = 3) -> str:
        parts: list[str] = [_BASE_OPENING]

        active = set(self.active_keys)
        if not active:
            parts.append("ruler-straight jawline and smooth neck")
        else:
            count = 0
            for key in PRIORITY_ORDER:
                if key not in active or key not in RHYTID_PROCEDURES:
                    continue
                proc = RHYTID_PROCEDURES[key]
                prefix = SEVERITY_PREFIX.get(self.get_severity(key), "")
                fragment = proc.prompt_fragment
                if prefix and not fragment.lower().startswith(prefix.strip().lower()):
                    fragment = prefix + fragment
                parts.append(fragment)
                count += 1
                if count >= max_procedures:
                    break

            for key in self.inactive_keys:
                parts.append(RHYTID_PROCEDURES[key].anchor_fragment)

        parts.append(_BASE_CLOSING)
        return ", ".join(parts)


def analyze_rhytidectomy(landmarks: FaceLandmarks) -> RhytidAnalysis:
    """Auto-detect which of the 8 rhytidectomy presets apply."""
    jaw = measure_jaw(landmarks)
    pts = landmarks.points
    _, h = landmarks.image_size

    detected: dict[str, bool] = {}
    sev: dict[str, int] = {}
    measurements = {"jaw_width": jaw["jaw_width"], "chin_y": jaw["chin_y"]}

    jaw_mean_y = jaw["jaw_mean_y"]
    chin_y = jaw["chin_y"]
    jaw_sag = float(chin_y - jaw_mean_y)
    measurements["jaw_sag"] = jaw_sag

    detected["jawline_straightening"] = True
    sev["jawline_straightening"] = Severity.MODERATE

    sag_is_prominent = jaw_sag > h * 0.08
    detected["jowl_elimination"] = sag_is_prominent
    if sag_is_prominent:
        sev["jowl_elimination"] = (Severity.SEVERE if jaw_sag > h * 0.14 else
                                    Severity.MODERATE if jaw_sag > h * 0.10 else Severity.MILD)

    detected["prejowl_correction"] = sag_is_prominent
    if sag_is_prominent:
        sev["prejowl_correction"] = sev["jowl_elimination"]

    mouth_left = pts[61] if 61 < len(pts) else pts[0]
    jaw_left = pts[172] if 172 < len(pts) else pts[0]
    marionette_depth = float(jaw_left[1] - mouth_left[1])
    measurements["marionette_depth"] = marionette_depth
    mar_active = marionette_depth > h * 0.05
    detected["marionette_softening"] = mar_active
    if mar_active:
        sev["marionette_softening"] = (Severity.SEVERE if marionette_depth > h * 0.09 else
                                        Severity.MODERATE if marionette_depth > h * 0.07 else Severity.MILD)

    neck_extent = h - chin_y
    has_neck = neck_extent > h * 0.1
    measurements["neck_extent_ratio"] = float(neck_extent / max(h, 1))
    detected["neck_smoothing"] = has_neck
    detected["platysmal_band_removal"] = has_neck
    detected["submental_definition"] = has_neck
    if has_neck:
        sev["neck_smoothing"] = Severity.MODERATE
        sev["platysmal_band_removal"] = Severity.MODERATE
        sev["submental_definition"] = Severity.MODERATE

    detected["nasolabial_softening"] = detected["jowl_elimination"]
    if detected["nasolabial_softening"]:
        sev["nasolabial_softening"] = sev["jowl_elimination"]

    n_active = sum(1 for v in detected.values() if v)
    analysis = RhytidAnalysis(detected=detected, severity=sev, measurements=measurements)
    log.info("Rhytid analysis: active=%d/%d (%s), jaw_sag=%.1f", n_active,
             len(detected), ", ".join(analysis.active_keys[:4]), jaw_sag)
    return analysis


def make_analysis(active: set[str], severities: dict[str, int] | None = None) -> RhytidAnalysis:
    """Surgeon-override factory: build an Analysis from explicit preset set."""
    unknown = active - set(RHYTID_PROCEDURES)
    if unknown:
        raise KeyError(f"Unknown rhytid preset keys: {sorted(unknown)}")

    detected = {k: (k in active) for k in RHYTID_PROCEDURES}
    sev = {k: Severity.MODERATE for k in active}
    if severities:
        sev.update({k: v for k, v in severities.items() if k in active})

    return RhytidAnalysis(detected=detected, severity=sev, measurements={})
