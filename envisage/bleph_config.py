"""Blepharoplasty sub-procedure taxonomy and auto-detection.

8 frontal-detectable presets. Each preset carries a positive
`prompt_fragment` (what changes when applied) and an `anchor_fragment`
(what stays identical when NOT applied). The anchor clauses are
load-bearing for preset-conditional editing: they pin non-requested
anatomy explicitly in the prompt.

Presets:
  1. upper_skin_excision    : remove redundant upper-lid skin
  2. crease_restoration     : define the supratarsal crease
  3. upper_dehooding        : lift hooded upper-lid fold
  4. lid_symmetry           : correct asymmetric crease height
  5. fat_pad_reduction      : upper medial fat-pad reduction
  6. lower_bag_reduction    : lower-lid fat/bag reduction
  7. tear_trough_smoothing  : smooth the lid-cheek junction
  8. crow_feet_softening    : soften lateral periorbital lines

Mask rules (upstream, masks.py):
- Upper-lid fold ONLY -- never iris/sclera/lashes
- Tight blend, low feather
- Preserve iris color
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from .landmarks import FaceLandmarks, measure_eyelid_hooding

log = logging.getLogger(__name__)


class BlephZone(str, Enum):
    UPPER = "upper_lid"
    LOWER = "lower_lid"


@dataclass(frozen=True)
class BlephSubProcedure:
    id: int
    key: str
    label: str
    zone: BlephZone
    prompt_fragment: str
    anchor_fragment: str
    measurement_key: str
    expected_sign: int          # +1 hooding increases (i.e. improves), -1 decreases, 0 texture-only
    delta_threshold: float      # baseline atlas-magnitude for MODERATE severity
    detectable_frontal: bool
    description: str


PRIORITY_ORDER = [
    "upper_skin_excision",
    "crease_restoration",
    "upper_dehooding",
    "lid_symmetry",
    "fat_pad_reduction",
    "lower_bag_reduction",
    "tear_trough_smoothing",
    "crow_feet_softening",
]


BLEPH_PROCEDURES: dict[str, BlephSubProcedure] = {
    "upper_skin_excision": BlephSubProcedure(
        id=1, key="upper_skin_excision",
        label="Upper Lid Skin Excision",
        zone=BlephZone.UPPER,
        prompt_fragment="tighter upper eyelid skin without excess folds or drooping",
        anchor_fragment="same upper eyelid skin amount as input",
        measurement_key="hooding_min",
        expected_sign=+1,
        delta_threshold=0.15,
        detectable_frontal=True,
        description="#1 most common. Remove redundant upper-lid skin (dermatochalasis).",
    ),
    "crease_restoration": BlephSubProcedure(
        id=2, key="crease_restoration",
        label="Supratarsal Crease Restoration",
        zone=BlephZone.UPPER,
        prompt_fragment="visible well-defined symmetric supratarsal crease on both upper eyelids",
        anchor_fragment="same supratarsal crease depth and position as input",
        measurement_key="hooding_min",
        expected_sign=+1,
        delta_threshold=0.15,
        detectable_frontal=True,
        description="Restore or create defined upper-lid crease. Canonical: 8-10mm female, 6-8mm male.",
    ),
    "upper_dehooding": BlephSubProcedure(
        id=3, key="upper_dehooding",
        label="Upper Lid De-hooding",
        zone=BlephZone.UPPER,
        prompt_fragment="refreshed upper eyelids with visible tarsal platform and no hooding",
        anchor_fragment="same upper-lid hooding as input",
        measurement_key="hooding_min",
        expected_sign=+1,
        delta_threshold=0.15,
        detectable_frontal=True,
        description="Remove excess skin covering the lid fold. Most visible improvement.",
    ),
    "lid_symmetry": BlephSubProcedure(
        id=4, key="lid_symmetry",
        label="Eyelid Symmetry Correction",
        zone=BlephZone.UPPER,
        prompt_fragment="symmetric bilateral eyelid creases at equal height",
        anchor_fragment="same left-right eyelid asymmetry as input",
        measurement_key="asymmetry",
        expected_sign=-1,
        delta_threshold=0.10,
        detectable_frontal=True,
        description="Correct asymmetric lid-crease height or hooding amount between eyes.",
    ),
    "fat_pad_reduction": BlephSubProcedure(
        id=5, key="fat_pad_reduction",
        label="Upper Medial Fat Pad Reduction",
        zone=BlephZone.UPPER,
        prompt_fragment="smooth upper eyelid contour without medial fat fullness",
        anchor_fragment="same upper-lid fullness and medial fat contour as input",
        measurement_key="hooding_min",
        expected_sign=+1,
        delta_threshold=0.10,
        detectable_frontal=True,
        description="Remove herniated medial fat pad causing upper-lid fullness.",
    ),
    "lower_bag_reduction": BlephSubProcedure(
        id=6, key="lower_bag_reduction",
        label="Lower Lid Bag Reduction",
        zone=BlephZone.LOWER,
        prompt_fragment="smooth lower eyelid contour without visible puffiness or bags",
        anchor_fragment="same lower-lid fullness as input",
        measurement_key="lower_bag",
        expected_sign=-1,
        delta_threshold=1.5,
        detectable_frontal=True,
        description="Remove or reposition herniated lower-lid fat pads.",
    ),
    "tear_trough_smoothing": BlephSubProcedure(
        id=7, key="tear_trough_smoothing",
        label="Tear Trough Smoothing",
        zone=BlephZone.LOWER,
        prompt_fragment="smooth lid-cheek junction without dark circles or tear trough depression",
        anchor_fragment="same tear-trough depth as input",
        measurement_key="lower_bag",
        expected_sign=-1,
        delta_threshold=1.0,
        detectable_frontal=True,
        description="Fill or smooth nasojugal groove via fat transposition or filler.",
    ),
    "crow_feet_softening": BlephSubProcedure(
        id=8, key="crow_feet_softening",
        label="Periorbital Fine-line Softening",
        zone=BlephZone.LOWER,
        prompt_fragment="softened periorbital fine lines with natural skin texture preserved",
        anchor_fragment="same lateral periorbital lines as input",
        measurement_key="hooding_min",
        expected_sign=+1,
        delta_threshold=0.10,
        detectable_frontal=True,
        description="Mild smoothing of lateral periorbital lines. Not full removal.",
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


_BASE_OPENING = (
    "a photorealistic frontal portrait of the same person, "
    "natural skin texture with visible pores"
)

# Bleph closing scaffold. The hardest failure mode is the upper lid
# losing its supratarsal crease and looking flat/featureless. The
# closing INSISTS on a visible crease, distinct fold edge, and natural
# lash-line, and explicitly forbids the flat-lid failure mode.
_BASE_CLOSING = (
    "clearly visible supratarsal crease on both upper eyelids, "
    "distinct well-defined fold line above the lash line, "
    "visible double eyelid crease with clear edge, "
    "no smooth featureless upper eyelid, "
    "no absent eyelid crease, "
    "no missing double fold, "
    "preserve iris and pupil exactly as in input, "
    "preserve lashes exactly as in input, "
    "both eyes identical and perfectly symmetric, "
    "identical eye color and iris color and sclera color, "
    "no dark patches no color drift on eyelid skin, "
    "sharp focus on lashes and iris, crisp eyelid edge, no blur, no softening, no smoothing artifacts, "
    "no hallucinated anatomy, no extra eyelid folds, no fabricated structures, "
    "studio lighting, high quality, photorealistic"
)


@dataclass
class BlephAnalysis:
    detected: dict[str, bool] = field(default_factory=dict)
    severity: dict[str, int] = field(default_factory=dict)
    measurements: dict[str, float] = field(default_factory=dict)

    @property
    def active_procedures(self) -> list[BlephSubProcedure]:
        return [BLEPH_PROCEDURES[k] for k, v in self.detected.items() if v]

    @property
    def active_keys(self) -> list[str]:
        return [k for k, v in self.detected.items() if v]

    @property
    def inactive_keys(self) -> list[str]:
        active = set(self.active_keys)
        return [k for k in BLEPH_PROCEDURES if k not in active]

    def get_severity(self, key: str) -> int:
        return self.severity.get(key, Severity.MODERATE)

    def build_prompt(self, max_procedures: int = 3) -> str:
        parts: list[str] = [_BASE_OPENING]

        active = set(self.active_keys)
        if not active:
            parts.append("refreshed eyelids with visible supratarsal crease on both eyes")
        else:
            count = 0
            for key in PRIORITY_ORDER:
                if key not in active or key not in BLEPH_PROCEDURES:
                    continue
                proc = BLEPH_PROCEDURES[key]
                prefix = SEVERITY_PREFIX.get(self.get_severity(key), "")
                fragment = proc.prompt_fragment
                if prefix and not fragment.lower().startswith(prefix.strip().lower()):
                    fragment = prefix + fragment
                parts.append(fragment)
                count += 1
                if count >= max_procedures:
                    break

            for key in self.inactive_keys:
                parts.append(BLEPH_PROCEDURES[key].anchor_fragment)

        parts.append(_BASE_CLOSING)
        return ", ".join(parts)


def analyze_blepharoplasty(landmarks: FaceLandmarks) -> BlephAnalysis:
    """Auto-detect which of the 8 blepharoplasty presets apply."""
    hooding = measure_eyelid_hooding(landmarks)
    pts = landmarks.points

    detected: dict[str, bool] = {}
    sev: dict[str, int] = {}
    measurements = {
        "left_hooding": hooding["left_hooding"],
        "right_hooding": hooding["right_hooding"],
        "asymmetry": hooding["asymmetry"],
    }

    min_hood = min(hooding["left_hooding"], hooding["right_hooding"])
    measurements["hooding_min"] = min_hood

    def _sev_hooding(threshold_activate: float) -> int:
        # Lower hooding = more hooded = more severe.
        if min_hood <= threshold_activate - 0.6:
            return Severity.SEVERE
        if min_hood <= threshold_activate - 0.3:
            return Severity.MODERATE
        if min_hood < threshold_activate:
            return Severity.MILD
        return Severity.NONE

    detected["upper_skin_excision"] = min_hood < 1.8
    if detected["upper_skin_excision"]:
        sev["upper_skin_excision"] = _sev_hooding(1.8)

    detected["crease_restoration"] = min_hood < 1.3
    if detected["crease_restoration"]:
        sev["crease_restoration"] = _sev_hooding(1.3)

    detected["upper_dehooding"] = min_hood < 1.5
    if detected["upper_dehooding"]:
        sev["upper_dehooding"] = _sev_hooding(1.5)

    detected["lid_symmetry"] = hooding["asymmetry"] > 0.3
    if detected["lid_symmetry"]:
        a = hooding["asymmetry"]
        sev["lid_symmetry"] = (Severity.SEVERE if a > 0.6 else
                               Severity.MODERATE if a > 0.45 else Severity.MILD)

    detected["fat_pad_reduction"] = min_hood < 1.2
    if detected["fat_pad_reduction"]:
        sev["fat_pad_reduction"] = _sev_hooding(1.2)

    left_lower = pts[145] if 145 < len(pts) else pts[0]
    right_lower = pts[374] if 374 < len(pts) else pts[0]
    left_cheek = pts[116] if 116 < len(pts) else pts[0]
    right_cheek = pts[345] if 345 < len(pts) else pts[0]
    left_bag = float(abs(left_lower[1] - left_cheek[1]))
    right_bag = float(abs(right_lower[1] - right_cheek[1]))
    measurements["left_lower_fullness"] = left_bag
    measurements["right_lower_fullness"] = right_bag
    measurements["lower_bag"] = min(left_bag, right_bag)

    detected["lower_bag_reduction"] = min(left_bag, right_bag) < 15
    if detected["lower_bag_reduction"]:
        v = min(left_bag, right_bag)
        sev["lower_bag_reduction"] = (Severity.SEVERE if v < 10 else
                                       Severity.MODERATE if v < 13 else Severity.MILD)

    detected["tear_trough_smoothing"] = min(left_bag, right_bag) < 12
    if detected["tear_trough_smoothing"]:
        v = min(left_bag, right_bag)
        sev["tear_trough_smoothing"] = (Severity.SEVERE if v < 8 else
                                         Severity.MODERATE if v < 10 else Severity.MILD)

    detected["crow_feet_softening"] = min_hood < 2.0
    if detected["crow_feet_softening"]:
        sev["crow_feet_softening"] = _sev_hooding(2.0)

    n_active = sum(1 for v in detected.values() if v)
    analysis = BlephAnalysis(detected=detected, severity=sev, measurements=measurements)
    log.info("Bleph analysis: active=%d/%d (%s)", n_active, len(detected),
             ", ".join(analysis.active_keys[:4]))
    return analysis


def make_analysis(active: set[str], severities: dict[str, int] | None = None) -> BlephAnalysis:
    """Surgeon-override factory: build an Analysis from explicit preset set."""
    unknown = active - set(BLEPH_PROCEDURES)
    if unknown:
        raise KeyError(f"Unknown bleph preset keys: {sorted(unknown)}")

    detected = {k: (k in active) for k in BLEPH_PROCEDURES}
    sev = {k: Severity.MODERATE for k in active}
    if severities:
        sev.update({k: v for k, v in severities.items() if k in active})

    return BlephAnalysis(detected=detected, severity=sev, measurements={})
