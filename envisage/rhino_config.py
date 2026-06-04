"""Rhinoplasty sub-procedure taxonomy and auto-detection.

8 frontal-detectable presets covering the dominant clinical intent of
rhinoplasty. Grounded in Rollin K. Daniel, "Mastering Rhinoplasty"
(2nd Ed). Each preset is visually distinct, landmark-detectable,
severity-scaled, and carries both a positive `prompt_fragment` (what
changes when the preset is applied) and an `anchor_fragment` (what
must stay the same when the preset is NOT applied).

The anchor_fragment is load-bearing for preset-conditional editing:
when the surgeon selects a subset of presets, the remaining presets'
anchor_fragments are emitted as explicit preservation clauses in the
prompt, preventing ambient drift into non-requested territory.

Presets (by clinical frequency):
  1. dorsal_hump_reduction   : ~70% of patients. #1 request.
  2. tip_definition          : "as the tip goes so goes the result"
  3. tip_narrowing           : narrower tip dome width
  4. dorsal_narrowing        : wide bony vault correction
  5. tip_rotation_up         : drooping/ptotic tip
  6. alar_base_narrowing     : wide nostril correction
  7. dorsal_straightening    : deviated/crooked dorsum
  8. nose_shortening         : long nose, caudal septal resection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .landmarks import (
    FaceLandmarks,
    measure_nose,
    measure_nasal_symmetry,
    NOSE_DORSUM,
)

log = logging.getLogger(__name__)


class RhinoZone(str, Enum):
    DORSUM = "dorsum"
    TIP_SHAPE = "tip_shape"
    TIP_POSITION = "tip_position"
    ALAR = "alar"


@dataclass(frozen=True)
class RhinoSubProcedure:
    """A discrete rhinoplasty sub-procedure with control metadata."""

    id: int                     # Daniel reference number
    key: str
    label: str
    zone: RhinoZone
    prompt_fragment: str        # positive: "straight dorsum aligned with midline"
    anchor_fragment: str        # negative/anchor: "same dorsal line as input"
    depth_action: str           # key into depth-modification registry
    measurement_key: str        # which landmark measurement this preset drives
    expected_sign: int          # direction of expected delta (+1 increase, -1 decrease, 0 texture-only)
    delta_threshold: float      # baseline atlas-magnitude for MODERATE severity (absolute units of the measurement)
    detectable_frontal: bool
    description: str


# Prompt builder emits positive fragments in this order (most impactful first)
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


RHINO_PROCEDURES: dict[str, RhinoSubProcedure] = {
    "dorsal_hump_reduction": RhinoSubProcedure(
        id=1, key="dorsal_hump_reduction",
        label="Dorsal Hump Removal",
        zone=RhinoZone.DORSUM,
        prompt_fragment="smooth straight dorsal profile with no visible hump or convexity on the bridge",
        anchor_fragment="same dorsal profile as input with identical bridge convexity",
        depth_action="flatten_bridge_convexity",
        measurement_key="bridge_x_spread",
        expected_sign=-1,
        delta_threshold=1.0,
        detectable_frontal=True,
        description="#1 most common request (~70% of patients). Bone rasp plus cartilage scissors.",
    ),
    "tip_definition": RhinoSubProcedure(
        id=13, key="tip_definition",
        label="Tip Definition",
        zone=RhinoZone.TIP_SHAPE,
        prompt_fragment="well-defined nasal tip with visible tip-defining points and supratip break",
        anchor_fragment="same tip shape as input with identical supratip region",
        depth_action="sharpen_tip_definition",
        measurement_key="tip_bulbosity",
        expected_sign=-1,
        delta_threshold=0.04,
        detectable_frontal=True,
        description="'As the tip goes so goes the result.' Domal suture plus tip refinement grafts.",
    ),
    "tip_narrowing": RhinoSubProcedure(
        id=12, key="tip_narrowing",
        label="Tip Narrowing",
        zone=RhinoZone.TIP_SHAPE,
        prompt_fragment="narrower nasal tip lobule with reduced dome width",
        anchor_fragment="same tip width as input",
        depth_action="narrow_tip_width",
        measurement_key="tip_bulbosity",
        expected_sign=-1,
        delta_threshold=0.04,
        detectable_frontal=True,
        description="Reduce dome width. L1 interdomal suture, L2 domal creation, L3 dome excision.",
    ),
    "dorsal_narrowing": RhinoSubProcedure(
        id=3, key="dorsal_narrowing",
        label="Bridge Narrowing",
        zone=RhinoZone.DORSUM,
        prompt_fragment="narrower refined nasal bridge with parallel dorsal aesthetic lines",
        anchor_fragment="same bridge width as input",
        depth_action="narrow_bony_vault",
        measurement_key="bridge_width_ratio",
        expected_sign=-1,
        delta_threshold=0.04,
        detectable_frontal=True,
        description="Lateral osteotomies narrow the bony vault. Indicated when alar-width to intercanthal-distance ratio exceeds 1.05.",
    ),
    "tip_rotation_up": RhinoSubProcedure(
        id=22, key="tip_rotation_up",
        label="Tip Rotation (Upward)",
        zone=RhinoZone.TIP_POSITION,
        prompt_fragment="slightly upturned nasal tip with natural nasolabial angle and supratip break",
        anchor_fragment="same tip projection angle as input",
        depth_action="rotate_tip_superiorly",
        measurement_key="tip_droop",
        expected_sign=-1,
        delta_threshold=4.0,
        detectable_frontal=True,
        description="Correct drooping or ptotic tip. Tip-position suture plus caudal septal resection.",
    ),
    "alar_base_narrowing": RhinoSubProcedure(
        id=35, key="alar_base_narrowing",
        label="Alar Base Narrowing",
        zone=RhinoZone.ALAR,
        prompt_fragment="narrower alar base with proportional nostril width",
        anchor_fragment="same alar base width and nostril width as input",
        depth_action="narrow_alar_base",
        measurement_key="alar_width",
        expected_sign=-1,
        delta_threshold=2.0,
        detectable_frontal=True,
        description="Reduce wide nostril base. Alar width should approximate intercanthal distance.",
    ),
    "dorsal_straightening": RhinoSubProcedure(
        id=4, key="dorsal_straightening",
        label="Dorsal Straightening",
        zone=RhinoZone.DORSUM,
        prompt_fragment="perfectly straight centered nasal dorsum aligned with facial midline",
        anchor_fragment="same dorsal midline alignment as input",
        depth_action="straighten_dorsal_line",
        measurement_key="dorsal_deviation_std",
        expected_sign=-1,
        delta_threshold=1.0,
        detectable_frontal=True,
        description="Asymmetric osteotomies plus septal relocation correct a crooked or deviated dorsum.",
    ),
    "nose_shortening": RhinoSubProcedure(
        id=46, key="nose_shortening",
        label="Nose Shortening",
        zone=RhinoZone.TIP_POSITION,
        prompt_fragment="shorter nasal length with proportional tip position",
        anchor_fragment="same nose length and tip vertical position as input",
        depth_action="shorten_nose",
        measurement_key="nose_length_ratio",
        expected_sign=-1,
        delta_threshold=0.03,
        detectable_frontal=True,
        description="Caudal septal resection plus tip rotation for an over-long nose.",
    ),
}


class Severity:
    """Sub-procedure severity levels with TPS/depth scaling."""

    NONE = 0
    MILD = 1
    MODERATE = 2
    SEVERE = 3


SEVERITY_PREFIX: dict[int, str] = {
    Severity.NONE: "",
    Severity.MILD: "slightly ",
    Severity.MODERATE: "",        # baseline; no prefix
    Severity.SEVERE: "pronounced ",
}


# Words that are incompatible with a severity-prefix. "slightly perfectly"
# is ungrammatical; strip the absolute and replace with the softer prefix.
_ABSOLUTES_TO_SOFTEN = ("perfectly ", "completely ", "fully ", "ruler-straight ")


def _compose_severity(prefix: str, fragment: str) -> str:
    """Compose a severity prefix with a positive fragment, handling grammar.

    - MODERATE (prefix == "") returns the fragment unchanged.
    - MILD / SEVERE strip any absolute qualifier in the fragment first so
      the output reads naturally ("slightly straight" not "slightly perfectly straight").
    """
    if not prefix:
        return fragment

    # Strip leading absolutes if present (case-insensitive)
    out = fragment
    lower = out.lower()
    for absolute in _ABSOLUTES_TO_SOFTEN:
        if lower.startswith(absolute):
            out = out[len(absolute):]
            lower = out.lower()
            break
        # also strip if the absolute appears early in the fragment
        idx = lower.find(" " + absolute)
        if 0 <= idx < 20:  # only if near the start
            out = out[:idx] + out[idx + len(absolute):]
            lower = out.lower()
            break

    # Avoid double-prefixing
    if lower.startswith(prefix.strip().lower()):
        return out
    return prefix + out


# Shared scaffold fragments (reused by all rhino prompts)
_BASE_OPENING = (
    "a photorealistic frontal portrait of the same person, "
    "natural skin texture with visible pores"
)
# Closing scaffold. The explicit anti-hallucination clauses target the
# known rhino failure modes: nostril-black holes, color drift inside the
# mask, melted or duplicated nostrils, and ambient skin-tone shifts.
_BASE_CLOSING = (
    "well-defined nostrils with clear nostril rim contour and natural soft shadows inside the nostrils, "
    "no dark holes, no black patches, no discolored regions, "
    "no melted or duplicated anatomy, no doubled features, "
    "sharp focus on all facial features, crisp skin texture, no blur, no softening, no smoothing artifacts, "
    "no hallucinated anatomy, no extra features, no fabricated structures, "
    "matching original lighting direction and shadow placement exactly, "
    "preserve existing moles and blemishes exactly as they are, "
    "identical skin tone and skin color as the original face, "
    "natural skin color continuous with the rest of the face, "
    "clinical photography lighting, high quality, photorealistic"
)


@dataclass
class RhinoAnalysis:
    """Result of automatic rhinoplasty sub-procedure detection."""

    detected: dict[str, bool] = field(default_factory=dict)
    severity: dict[str, int] = field(default_factory=dict)
    measurements: dict[str, float] = field(default_factory=dict)
    level: int = 1

    @property
    def active_procedures(self) -> list[RhinoSubProcedure]:
        return [RHINO_PROCEDURES[k] for k, v in self.detected.items() if v]

    @property
    def active_keys(self) -> list[str]:
        return [k for k, v in self.detected.items() if v]

    @property
    def inactive_keys(self) -> list[str]:
        active = set(self.active_keys)
        return [k for k in RHINO_PROCEDURES if k not in active]

    def get_severity(self, key: str) -> int:
        return self.severity.get(key, Severity.NONE)

    def tps_scale(self, key: str) -> float:
        """TPS displacement multiplier based on severity. 0.0 to 1.5."""
        s = self.get_severity(key)
        return {Severity.NONE: 0.0, Severity.MILD: 0.5,
                Severity.MODERATE: 1.0, Severity.SEVERE: 1.5}.get(s, 0.0)

    def build_prompt(self, max_procedures: int = 3) -> str:
        """Compose positive (active presets) + anchor (inactive presets) + scaffold.

        Positive fragments are severity-prefixed ("slightly", "", "pronounced")
        and capped at `max_procedures`. Anchor fragments are emitted for every
        inactive preset, with no severity modulation, to pin non-requested
        anatomy explicitly.

        Returns a single comma-joined string suitable for FLUX guidance.
        """
        parts: list[str] = [_BASE_OPENING]

        active = set(self.active_keys)

        if not active:
            # No procedure selected; emit generic scaffold.
            parts.append("refined symmetric nose with straight bridge and defined tip")
        else:
            # Positive fragments for top-N active presets, in priority order.
            count = 0
            for key in PRIORITY_ORDER:
                if key not in active or key not in RHINO_PROCEDURES:
                    continue
                proc = RHINO_PROCEDURES[key]
                prefix = SEVERITY_PREFIX.get(self.get_severity(key), "")
                fragment = _compose_severity(prefix, proc.prompt_fragment)
                parts.append(fragment)
                count += 1
                if count >= max_procedures:
                    break

            # Anchor fragments for all inactive presets (explicit preservation).
            for key in self.inactive_keys:
                proc = RHINO_PROCEDURES[key]
                parts.append(proc.anchor_fragment)

        parts.append(_BASE_CLOSING)
        return ", ".join(parts)

    def build_depth_plan(self) -> list[str]:
        """Ordered list of depth-modification actions for active presets."""
        active = set(self.active_keys)
        return [
            RHINO_PROCEDURES[k].depth_action
            for k in PRIORITY_ORDER
            if k in active and k in RHINO_PROCEDURES
        ]

    def summary(self) -> str:
        active = self.active_procedures
        if not active:
            return "No procedures detected"
        lines = [f"Level {self.level} rhinoplasty ({len(active)} procedures):"]
        for p in active:
            sev = self.get_severity(p.key)
            sev_label = {0: "NONE", 1: "MILD", 2: "MOD", 3: "SEV"}.get(sev, "?")
            lines.append(f"  [{p.zone.value}] #{p.id} {p.label} ({sev_label})")
        return "\n".join(lines)


def _severity_from_thresholds(value: float, thresholds: tuple[float, float, float]) -> int:
    """Map an absolute magnitude to MILD/MODERATE/SEVERE via 3 thresholds."""
    mild, moderate, severe = thresholds
    if value >= severe:
        return Severity.SEVERE
    if value >= moderate:
        return Severity.MODERATE
    if value >= mild:
        return Severity.MILD
    return Severity.NONE


def analyze_rhinoplasty(landmarks: FaceLandmarks) -> RhinoAnalysis:
    """Auto-detect which of the 8 rhinoplasty presets apply.

    Uses Daniel measurement criteria on frontal-view MediaPipe landmarks.
    Emits severity (MILD/MODERATE/SEVERE) per detected preset.
    """
    nose = measure_nose(landmarks)
    sym = measure_nasal_symmetry(landmarks)
    pts = landmarks.points
    w, _ = landmarks.image_size

    detected: dict[str, bool] = {}
    sev: dict[str, int] = {}
    measurements = {
        "alar_width": sym["alar_width"],
        "intercanthal_distance": sym["intercanthal_distance"],
        "bridge_width_ratio": sym["bridge_width_ratio"],
        "tip_bulbosity": sym["tip_bulbosity"],
        "dorsal_deviation_std": sym["dorsal_deviation_std"],
    }

    bwr = sym["bridge_width_ratio"]
    bulb = sym["tip_bulbosity"]
    dev_std = sym["dorsal_deviation_std"]

    bridge_pts = pts[[i for i in NOSE_DORSUM if i < len(pts)]]
    bridge_x_spread = float(np.std(bridge_pts[:, 0])) if len(bridge_pts) > 3 else 0.0
    measurements["bridge_x_spread"] = bridge_x_spread
    detected["dorsal_hump_reduction"] = bridge_x_spread > 2.5
    if detected["dorsal_hump_reduction"]:
        sev["dorsal_hump_reduction"] = _severity_from_thresholds(bridge_x_spread, (2.5, 3.5, 5.0))

    hump_causes_widening = detected["dorsal_hump_reduction"] and bridge_x_spread < 5.0
    measurements["hump_causes_widening"] = 1.0 if hump_causes_widening else 0.0

    detected["dorsal_narrowing"] = bwr > 1.05 and not hump_causes_widening
    if detected["dorsal_narrowing"]:
        sev["dorsal_narrowing"] = _severity_from_thresholds(bwr, (1.05, 1.10, 1.15))

    detected["dorsal_straightening"] = dev_std > 2.0
    if detected["dorsal_straightening"]:
        sev["dorsal_straightening"] = _severity_from_thresholds(dev_std, (2.0, 3.0, 4.0))

    detected["tip_narrowing"] = bulb > 0.45
    if detected["tip_narrowing"]:
        sev["tip_narrowing"] = _severity_from_thresholds(bulb, (0.45, 0.50, 0.60))

    detected["tip_definition"] = bulb > 0.40
    if detected["tip_definition"]:
        sev["tip_definition"] = _severity_from_thresholds(bulb, (0.40, 0.50, 0.55))

    tip = pts[1] if 1 < len(pts) else pts[0]
    subnasale = pts[2] if 2 < len(pts) else pts[0]
    tip_droop = float(tip[1] - subnasale[1])
    measurements["tip_droop"] = tip_droop
    detected["tip_rotation_up"] = tip_droop > 5.0
    if detected["tip_rotation_up"]:
        sev["tip_rotation_up"] = _severity_from_thresholds(tip_droop, (5.0, 7.0, 10.0))

    detected["alar_base_narrowing"] = bwr > 1.05
    if detected["alar_base_narrowing"]:
        sev["alar_base_narrowing"] = _severity_from_thresholds(bwr, (1.05, 1.10, 1.15))

    nose_length_ratio = nose["height"] / max(w, 1)
    measurements["nose_length_ratio"] = float(nose_length_ratio)
    detected["nose_shortening"] = nose_length_ratio > 0.28
    if detected["nose_shortening"]:
        sev["nose_shortening"] = _severity_from_thresholds(nose_length_ratio, (0.28, 0.32, 0.36))

    n_active = sum(1 for v in detected.values() if v)
    level = 1 if n_active <= 3 else (2 if n_active <= 5 else 3)

    analysis = RhinoAnalysis(detected=detected, severity=sev, measurements=measurements, level=level)
    log.info(
        "Rhino analysis: level=%d, active=%d/%d (%s)",
        level, n_active, len(detected),
        ", ".join(analysis.active_keys),
    )
    return analysis


def make_analysis(active: set[str], severities: dict[str, int] | None = None) -> RhinoAnalysis:
    """Surgeon-override factory: build an Analysis from an explicit preset set.

    Used when a surgeon selects presets at inference time rather than letting
    the auto-analyzer derive them. Severity defaults to MODERATE when unset.
    """
    unknown = active - set(RHINO_PROCEDURES)
    if unknown:
        raise KeyError(f"Unknown rhino preset keys: {sorted(unknown)}")

    detected = {k: (k in active) for k in RHINO_PROCEDURES}
    sev = {k: Severity.MODERATE for k in active}
    if severities:
        sev.update({k: v for k, v in severities.items() if k in active})

    return RhinoAnalysis(detected=detected, severity=sev, measurements={}, level=1)
