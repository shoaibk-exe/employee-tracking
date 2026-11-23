"""Structured VLM appearance features for hybrid ReID+VLM person matching.

Always extracts the same fixed keys. Values are lowercase tokens or ``unknown``.
When comparing two people, any key that is ``unknown`` on either side is skipped.

Parsing is intentionally strict: ambiguous free-text (prompt echoes, mixed
yes/no phrases, conflicting colors) becomes ``unknown`` rather than a guess.

Backends (see ``create_feature_extractor``):
* Moondream2 — small/fast; one attribute at a time
* Qwen2.5-VL — stronger colours on CCTV; same crop + parse pipeline

Accessory attributes (glasses / hat / hair) use a head-region crop;
clothing colours use torso / lower ROIs; gender uses the full person crop.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Fixed schema — always asked, always stored, always compared (unless unknown).
FEATURE_KEYS: tuple[str, ...] = (
    "gender",       # male | female | unknown
    "glasses",      # yes | no | unknown
    "top_color",    # color token | unknown
    "bottom_color", # color token | unknown
    "top_pattern",  # solid | plaid | other | unknown
    "hair",         # short | long | bald | unknown
    "hat",          # yes | no | unknown
)

# Queried on the head-region crop (need facial/head detail).
HEAD_ATTR_KEYS: tuple[str, ...] = ("glasses", "hat", "hair")
# Queried on the full person crop.
BODY_ATTR_KEYS: tuple[str, ...] = ("gender",)
# Queried on torso / lower ROIs (ignore head + neighbours when possible).
COLOR_ATTR_KEYS: tuple[str, ...] = ("top_color", "bottom_color")
# Queried on the torso ROI (shirt texture).
TORSO_ATTR_KEYS: tuple[str, ...] = ("top_pattern",)
# Keys that use the softer gallery vote share (noisy CCTV colours / patterns).
COLOR_FEATURE_KEYS: tuple[str, ...] = ("top_color", "bottom_color", "top_pattern")

UNKNOWN = "unknown"

# Evidence-oriented accessory prompts (validated on T27 / T22 CCTV crops).
# Direct "are they wearing X?" on skinny full-body crops false-positives hat/glasses.
# Colour prompts stay SHORT — a forced palette menu made Moondream answer "black"
# even on bright red shirts.
SINGLE_ATTR_PROMPTS: dict[str, str] = {
    "gender": (
        "Is the person male or female? "
        "Reply with one word only: male, female, or unknown."
    ),
    "glasses": (
        "Look at the person's eyes. Reply with one word only: yes, no, or unknown. "
        "yes = eyeglass frames or lenses are clearly visible. "
        "no = the eye area is clear enough to see and no eyeglasses are present. "
        "unknown = eyes are hidden, looking down, too blurry, or too small to judge."
    ),
    "top_color": (
        "What is the dominant color of this person's shirt or top? "
        "Ignore other people and background. If patterned, name the base color. "
        "Tan or khaki → beige. Navy → blue. Teal or olive → green. "
        "Blue-and-white plaid or check → blue. Green plaid or flannel → green. "
        "White, cream, or light gray → white or gray (never blue). "
        "Do not answer beige or brown for a dark shirt. "
        "Do not answer blue unless the fabric itself looks blue or navy. "
        "Reply with one color word only, or unknown."
    ),
    "bottom_color": (
        "What is the dominant color of this person's pants or lower clothing? "
        "Ignore other people. "
        "Answer blue only if the pants are clearly blue or denim; "
        "dark pants without a clear blue tint → black or gray. "
        "Reply with one color word only, or unknown."
    ),
    "top_pattern": (
        "Look at the shirt fabric only. "
        "plaid = repeating check, grid, or flannel squares (even if blurry). "
        "other = camo, floral, or irregular print. "
        "solid = one uniform color with no pattern. "
        "If unsure, unknown. Ignore other people. "
        "Reply with one word only: solid, plaid, other, or unknown."
    ),
    "hair": (
        "Is the person's hair short, long, or bald? "
        "Reply with one word only: short, long, bald, or unknown."
    ),
    "hat": (
        "Look only at the top of the head. Is a hat or cap clearly visible "
        "(not hair or shadow)? If the head is too small/blurry to tell, say unknown. "
        "Reply with one word only: yes, no, or unknown."
    ),
}

# Explicit color aliases → canonical name. Conflicting matches → unknown.
_COLOR_ALIASES: dict[str, str] = {
    "black": "black",
    "white": "white",
    "gray": "gray",
    "grey": "gray",
    "charcoal": "gray",
    "blue": "blue",
    "navy": "blue",
    "denim": "blue",
    "cyan": "blue",
    "red": "red",
    "maroon": "red",
    "pink": "pink",
    "green": "green",
    "olive": "green",
    "teal": "green",
    "yellow": "yellow",
    "mustard": "yellow",
    "ochre": "yellow",
    "orange": "orange",
    "brown": "brown",
    "beige": "beige",
    "tan": "beige",
    "khaki": "beige",
    "sand": "beige",
    "camel": "beige",
    "cream": "white",
    "purple": "purple",
    "violet": "purple",
    "slate": "blue",  # blue-gray office shirts → blue (chromatic wins over gray below)
    "forest": "green",
    "mint": "green",
    "turquoise": "green",
    "aqua": "blue",
}

# When a reply names gray/black plus one chromatic colour ("blue gray", "dark green"),
# keep the chromatic colour — office CCTV often hedges that way.
_CHROMATIC_COLORS = frozenset({
    "beige", "brown", "red", "orange", "yellow", "green", "blue", "purple", "pink",
})
_NEUTRAL_COLORS = frozenset({"black", "white", "gray"})


@dataclass
class VlmFeatureConfig:
    # Default stays Moondream for unit tests; production settings.yaml selects Qwen.
    model_id: str = "vikhyatk/moondream2"
    # Empty revision = HuggingFace default branch (required for Qwen checkpoints).
    revision: str = "2025-06-21"
    device: str = "auto"
    # Cap generation length (Qwen); Moondream ignores this.
    max_new_tokens: int = 24
    # Majority vote needs this many agreeing votes; else that feature → unknown.
    majority_min: int = 2
    # At least this many comparable (non-unknown) features must agree for a VLM match.
    min_comparable: int = 2
    # When set, every extract saves crop + raw reply under this directory.
    debug_directory: Path | None = None
    # Head region as a fraction of person-crop height (from the top).
    head_height_fraction: float = 0.38
    # Extra horizontal context inside the person crop when forming the head ROI.
    head_pad_x_fraction: float = 0.0
    # Minimum head-crop side (px). Below this, accessory attrs → unknown (no guess).
    min_head_side_px: int = 24
    # Torso / lower bands for clothing colour (fractions of person-crop height).
    torso_y0_fraction: float = 0.18
    torso_y1_fraction: float = 0.58
    lower_y0_fraction: float = 0.50
    lower_y1_fraction: float = 0.95
    # Colour crops below this side (px) or outside this mean-brightness band → unknown.
    min_color_side_px: int = 40
    min_color_mean: float = 22.0
    max_color_mean: float = 240.0
    # Clothing crops below this quality score do not vote colour/pattern.
    min_color_quality: float = 0.35


def empty_features() -> dict[str, str]:
    return {k: UNKNOWN for k in FEATURE_KEYS}


def crop_sha1(bgr_crop: np.ndarray) -> str:
    """Stable content hash for duplicate-crop rejection."""
    arr = np.ascontiguousarray(bgr_crop)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def expand_person_bbox(
    bbox: Sequence[float],
    frame_wh: tuple[int, int],
    *,
    pad_x_fraction: float = 0.15,
    pad_top_fraction: float = 0.08,
    pad_bottom_fraction: float = 0.0,
) -> tuple[int, int, int, int]:
    """Expand a person bbox for VLM context; clip to frame. Aspect preserved (no stretch)."""
    fw, fh = frame_wh
    x1, y1, x2, y2 = (float(v) for v in bbox)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    px = bw * float(pad_x_fraction)
    pt = bh * float(pad_top_fraction)
    pb = bh * float(pad_bottom_fraction)
    nx1 = int(max(0, np.floor(x1 - px)))
    ny1 = int(max(0, np.floor(y1 - pt)))
    nx2 = int(min(fw, np.ceil(x2 + px)))
    ny2 = int(min(fh, np.ceil(y2 + pb)))
    if nx2 <= nx1 or ny2 <= ny1:
        return (
            int(max(0, np.floor(x1))),
            int(max(0, np.floor(y1))),
            int(min(fw, np.ceil(x2))),
            int(min(fh, np.ceil(y2))),
        )
    return nx1, ny1, nx2, ny2


def head_crop_from_person(
    bgr_person: np.ndarray,
    *,
    height_fraction: float = 0.38,
    pad_x_fraction: float = 0.0,
) -> np.ndarray | None:
    """Top-of-person head ROI. Preserves aspect; does not upscale.

    ``bgr_person`` should already be taken from the source frame (optionally with
    modest bbox padding). Coordinates are relative to that crop.
    """
    if bgr_person is None or bgr_person.size == 0:
        return None
    h, w = bgr_person.shape[:2]
    if h < 2 or w < 2:
        return None
    frac = float(np.clip(height_fraction, 0.15, 0.6))
    hh = max(1, int(round(h * frac)))
    pad = int(round(w * max(0.0, float(pad_x_fraction))))
    x1 = max(0, 0 - pad)  # pad only meaningful if person crop already includes side context
    x2 = min(w, w + pad)
    # Within-crop: use full width (caller should pad the person crop from the frame).
    crop = bgr_person[0:hh, 0:w]
    if crop.size == 0:
        return None
    return np.ascontiguousarray(crop)


def head_crop_usable(bgr_head: np.ndarray | None, *, min_side_px: int = 24) -> bool:
    if bgr_head is None or bgr_head.size == 0:
        return False
    h, w = bgr_head.shape[:2]
    return min(h, w) >= int(min_side_px)


def _band_crop_from_person(
    bgr_person: np.ndarray,
    *,
    y0_fraction: float,
    y1_fraction: float,
) -> np.ndarray | None:
    """Horizontal band of a person crop (aspect preserved, no upscale)."""
    if bgr_person is None or bgr_person.size == 0:
        return None
    h, w = bgr_person.shape[:2]
    if h < 2 or w < 2:
        return None
    y0 = float(np.clip(y0_fraction, 0.0, 0.95))
    y1 = float(np.clip(y1_fraction, y0 + 0.05, 1.0))
    ya = max(0, int(round(h * y0)))
    yb = min(h, int(round(h * y1)))
    if yb <= ya:
        return None
    crop = bgr_person[ya:yb, 0:w]
    if crop.size == 0:
        return None
    return np.ascontiguousarray(crop)


def torso_crop_from_person(
    bgr_person: np.ndarray,
    *,
    y0_fraction: float = 0.18,
    y1_fraction: float = 0.58,
) -> np.ndarray | None:
    """Mid-body ROI for shirt / top colour (skips head and lower legs)."""
    return _band_crop_from_person(
        bgr_person, y0_fraction=y0_fraction, y1_fraction=y1_fraction,
    )


def lower_crop_from_person(
    bgr_person: np.ndarray,
    *,
    y0_fraction: float = 0.50,
    y1_fraction: float = 0.95,
) -> np.ndarray | None:
    """Lower-body ROI for pants / bottom colour."""
    return _band_crop_from_person(
        bgr_person, y0_fraction=y0_fraction, y1_fraction=y1_fraction,
    )


def clothing_crop_usable(
    bgr: np.ndarray | None,
    *,
    min_side_px: int = 40,
    min_mean: float = 22.0,
    max_mean: float = 240.0,
) -> bool:
    """Reject tiny / near-black / blown-out clothing crops (no colour guess)."""
    if bgr is None or bgr.size == 0:
        return False
    h, w = bgr.shape[:2]
    if min(h, w) < int(min_side_px):
        return False
    mean = float(np.mean(bgr))
    return float(min_mean) <= mean <= float(max_mean)


def clothing_crop_quality(bgr: np.ndarray | None) -> float:
    """0..1 score for how trustworthy a clothing crop is for colour/pattern.

    Prefers larger mid-brightness crops. Low scores should not vote into the gallery
    (human review uses a few clear frames, not every noisy one).
    """
    if bgr is None or bgr.size == 0:
        return 0.0
    h, w = bgr.shape[:2]
    side = float(min(h, w))
    size_s = max(0.0, min(1.0, (side - 24.0) / 56.0))  # 24→0, 80→1
    mean = float(np.mean(bgr))
    # Peak quality around typical shirt brightness; crush near-black / blown.
    bright_s = max(0.0, 1.0 - abs(mean - 115.0) / 115.0)
    return float(max(0.0, min(1.0, 0.55 * size_s + 0.45 * bright_s)))


def estimate_clothing_color_cv(bgr: np.ndarray | None) -> str | None:
    """Cheap HSV dominant-colour hint from a clothing crop.

    Returns a canonical colour or ``None`` when unsure. Used only to upgrade a
    VLM ``black`` / ``unknown`` (and tan ``brown``→``beige``) when the crop has
    a clear chromatic tint that Moondream often collapses to black.
    """
    if bgr is None or bgr.size == 0:
        return None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # Drop near-black / blown pixels; they dominate CCTV shadows and highlights.
    usable = (v >= 28) & (v <= 245)
    if int(usable.sum()) < 80:
        return None
    chromatic = usable & (s >= 45)
    n_chr = int(chromatic.sum())
    n_use = int(usable.sum())
    if n_chr >= max(60, int(0.12 * n_use)):
        hues = h[chromatic].astype(np.int32)
        # OpenCV hue is 0..179; 9 bins of 20.
        hist = np.bincount(hues // 20, minlength=9)
        peak = int(np.argmax(hist))
        # Map peak bin → colour (rough but stable for office shirts).
        bin_to_color = {
            0: "red",      # 0-19 ≈ red
            1: "orange",   # 20-39
            2: "yellow",   # 40-59
            3: "green",    # 60-79
            4: "green",    # 80-99 teal/cyan → green for clothing
            5: "blue",     # 100-119
            6: "blue",     # 120-139
            7: "purple",   # 140-159
            8: "red",      # 160-179 magenta/red
        }
        color = bin_to_color.get(peak)
        if color and hist[peak] >= max(40, int(0.35 * n_chr)):
            return color
    # Low-sat neutrals: beige vs gray vs brown from value + mild yellow hue.
    neutrals = usable & (s < 45)
    if int(neutrals.sum()) < 80:
        return None
    mean_v = float(v[neutrals].mean())
    mean_s = float(s[neutrals].mean())
    mean_h = float(h[neutrals].mean())
    if mean_v >= 140 and mean_s <= 55 and 10 <= mean_h <= 40:
        return "beige"
    if mean_v < 70:
        return "black"
    if mean_s <= 35:
        return "gray"
    if 5 <= mean_h <= 30 and mean_v < 130:
        return "brown"
    return None


def refine_color_with_cv(vlm_color: str, cv_color: str | None) -> str:
    """Upgrade weak VLM colours when CV sees a clear cool tint.

    Only rescues ``black`` / ``unknown`` → ``green`` / ``blue``. Warm upgrades
    (red/beige) are skipped — this CCTV feed has a red cast that fools HSV.
    """
    v = str(vlm_color or UNKNOWN)
    if cv_color is None:
        return v
    if v in {UNKNOWN, "black"} and cv_color in {"green", "blue"}:
        return cv_color
    return v


def sanitize_top_color(color: str, torso_bgr: np.ndarray | None) -> str:
    """Drop beige/brown on dark torsos; drop false blue on bright light shirts."""
    v = str(color or UNKNOWN)
    if torso_bgr is None or torso_bgr.size == 0:
        return v
    mean = float(np.mean(torso_bgr))
    # Dark office shirts (navy/maroon/teal) often get called beige under warm light.
    if v in {"beige", "brown"} and mean < 75.0:
        return UNKNOWN
    # Qwen often answers "blue" for white/cream/light-gray shirts under cool monitors.
    if v == "blue" and mean >= 145.0:
        cv = estimate_clothing_color_cv(torso_bgr)
        if cv in {"beige", "gray", "white"}:
            return cv
        if cv is None:
            hsv = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2HSV)
            sat = float(np.mean(hsv[:, :, 1]))
            if sat < 40.0:
                return "gray" if mean < 185.0 else "white"
    return v


def sanitize_bottom_color(color: str, lower_bgr: np.ndarray | None) -> str:
    """Drop blue on very dark lower crops unless a cool tint is visible."""
    v = str(color or UNKNOWN)
    if v != "blue" or lower_bgr is None or lower_bgr.size == 0:
        return v
    mean = float(np.mean(lower_bgr))
    if mean >= 55.0:
        return v
    # Near-black pants: keep blue only if CV also says blue.
    cv = estimate_clothing_color_cv(lower_bgr)
    return "blue" if cv == "blue" else "black"


# Warm accents often come from a neighbouring red tee in the same crop.
_WARM_TOP_COLORS = frozenset({"red", "pink", "orange"})
_COOL_TOP_COLORS = frozenset({
    "black", "white", "gray", "beige", "brown", "green", "blue", "yellow", "purple",
})


def parse_binary(raw: str) -> str:
    """Strict yes/no. Phrases like ``no, not wearing`` → unknown (never yes)."""
    value = (raw or "").strip().lower().strip(".,;:\"'`")
    if value in {"yes", "true"}:
        return "yes"
    if value in {"no", "false"}:
        return "no"
    # Extra words / commas / slashes → refuse to guess (classic 'wearing'→yes bug).
    return UNKNOWN


def parse_gender(raw: str) -> str:
    value = (raw or "").strip().lower().strip(".,;:\"'`")
    if " " in value:
        value = value.split()[0].strip(".,;:\"'`")
    if value in {"male", "female"}:
        return value
    return UNKNOWN


def parse_hair(raw: str) -> str:
    value = (raw or "").strip().lower().strip(".,;:\"'`")
    if " " in value:
        value = value.split()[0].strip(".,;:\"'`")
    if value in {"short", "long", "bald"}:
        return value
    return UNKNOWN


def parse_pattern(raw: str) -> str:
    """Map shirt pattern reply → solid | plaid | other | unknown."""
    value = (raw or "").strip().lower().strip(".,;:\"'`")
    if not value:
        return UNKNOWN
    # Take first token; allow short phrases like "plaid pattern".
    token = value.split()[0].strip(".,;:\"'`")
    if token in {"solid", "plain"}:
        return "solid"
    if token in {"plaid", "check", "checked", "checkered", "tartan", "flannel"}:
        return "plaid"
    if token in {"other", "patterned", "pattern", "print", "printed", "camo",
                 "camouflage", "floral", "striped", "stripe", "graphic"}:
        return "other"
    if "plaid" in value or "check" in value:
        return "plaid"
    if "solid" in value or "plain" in value:
        return "solid"
    return UNKNOWN


def parse_color(raw: str) -> str:
    """Map a colour reply to a canonical name.

    Bare ``dark`` / ``light`` alone → unknown. Phrases like ``dark green`` or
    ``blue gray`` keep the chromatic colour when paired with a neutral.
    Multiple distinct chromatics → unknown (conflict).
    """
    t = (raw or "").strip().lower()
    if not t or t in {"?", "n/a", "na", "none", "null", UNKNOWN, "dark", "light"}:
        return UNKNOWN
    found: list[str] = []
    for token in re.findall(r"[a-z]+", t):
        if token in {UNKNOWN, "dark", "light"}:
            continue
        canon = _COLOR_ALIASES.get(token)
        if canon is None:
            continue
        if canon not in found:
            found.append(canon)
    if not found:
        return UNKNOWN
    if len(found) == 1:
        return found[0]
    chromatics = [c for c in found if c in _CHROMATIC_COLORS]
    neutrals = [c for c in found if c in _NEUTRAL_COLORS]
    if len(chromatics) == 1 and neutrals:
        return chromatics[0]
    return UNKNOWN


def _canon_value(key: str, raw: str) -> str:
    if key in {"glasses", "hat"}:
        return parse_binary(raw)
    if key == "gender":
        return parse_gender(raw)
    if key == "hair":
        return parse_hair(raw)
    if key == "top_pattern":
        return parse_pattern(raw)
    if key in {"top_color", "bottom_color"}:
        return parse_color(raw)
    return UNKNOWN


def parse_feature_text(text: str) -> dict[str, str]:
    """Parse a multi-line VLM reply into the fixed feature dict (strict).

    Kept for tests and legacy; live extraction uses single-attribute answers.
    """
    out = empty_features()
    if not text:
        return out
    for line in text.replace("\r", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            key, _, val = line.partition(":")
        elif "=" in line:
            key, _, val = line.partition("=")
        else:
            continue
        key = key.strip().lower().replace(" ", "_").replace("-", "_")
        aliases = {
            "top": "top_color", "shirt_color": "top_color", "shirt": "top_color",
            "bottom": "bottom_color", "pants_color": "bottom_color", "pants": "bottom_color",
            "pattern": "top_pattern", "shirt_pattern": "top_pattern", "top_pat": "top_pattern",
            "hairstyle": "hair", "hair_style": "hair", "hair_length": "hair",
            "cap": "hat", "headwear": "hat",
        }
        key = aliases.get(key, key)
        if key not in FEATURE_KEYS:
            continue
        val_l = val.strip().lower()
        if (
            "," in val_l
            or "/" in val_l
            or "|" in val_l
            or any(tok in val_l for tok in (
                " or ", "allowed", "one color", "do not", "physical headwear",
                "headphones", "example", "repeat these",
            ))
        ):
            out[key] = UNKNOWN
            continue
        out[key] = _canon_value(key, val)
    return out


def parse_single_attr(key: str, raw: str) -> str:
    """Parse one attribute answer (strict). Echoed menus → unknown."""
    val_l = (raw or "").strip().lower()
    if (
        "," in val_l
        or "/" in val_l
        or "|" in val_l
        or " or " in val_l
        or val_l.count(" ") > 3
    ):
        return UNKNOWN
    return _canon_value(key, raw)


def majority_vote(
    samples: Sequence[Mapping[str, str]],
    *,
    majority_min: int = 2,
    allow_single: bool = True,
) -> dict[str, str]:
    """Per-feature majority across samples; minority / tie → unknown.

    ``allow_single=False``: a lone concrete observation is not trusted (identity
    decisions); it must reach ``majority_min`` like any other value.
    """
    out = empty_features()
    if not samples:
        return out
    for key in FEATURE_KEYS:
        votes = [str(s.get(key, UNKNOWN)) for s in samples]
        concrete = [v for v in votes if v != UNKNOWN]
        pool = concrete if concrete else votes
        if not pool:
            continue
        winner, count = Counter(pool).most_common(1)[0]
        # A single unique observation is still usable (static employees).
        need = majority_min
        if len(concrete) == 1 and allow_single:
            need = 1
        else:
            need = max(majority_min, (len(concrete) + 1) // 2 if concrete else majority_min)
        if winner != UNKNOWN and count >= need:
            out[key] = winner
        else:
            out[key] = UNKNOWN
    return out


def features_compatible(
    a: Mapping[str, str],
    b: Mapping[str, str],
    *,
    min_comparable: int = 2,
) -> tuple[bool, int, int]:
    """Return (ok, n_comparable, n_agree)."""
    comparable = agree = 0
    for key in FEATURE_KEYS:
        va = str(a.get(key, UNKNOWN))
        vb = str(b.get(key, UNKNOWN))
        if va == UNKNOWN or vb == UNKNOWN:
            continue
        comparable += 1
        if va == vb:
            agree += 1
        else:
            return False, comparable, agree
    if comparable < min_comparable:
        return False, comparable, agree
    return agree == comparable, comparable, agree


def feature_agreement(a: Mapping[str, str], b: Mapping[str, str]) -> tuple[int, int]:
    """Return (n_comparable, n_agree) over keys concrete on both sides."""
    comparable = agree = 0
    for key in FEATURE_KEYS:
        va = str(a.get(key, UNKNOWN))
        vb = str(b.get(key, UNKNOWN))
        if va == UNKNOWN or vb == UNKNOWN:
            continue
        comparable += 1
        if va == vb:
            agree += 1
    return comparable, agree


def add_feature_votes(votes: dict[str, Counter], sample: Mapping[str, str]) -> None:
    """Accumulate one sample's concrete values into per-key vote counters."""
    for key in FEATURE_KEYS:
        v = str(sample.get(key, UNKNOWN))
        if v != UNKNOWN:
            votes.setdefault(key, Counter())[v] += 1


def stable_features(
    votes: Mapping[str, Counter],
    current: Mapping[str, str] | None = None,
    *,
    min_votes: int = 2,
    min_share: float = 0.6,
    color_min_share: float | None = None,
    color_min_votes: int | None = None,
) -> dict[str, str]:
    """Gallery features from accumulated votes.

    A value is committed only with ``min_votes`` agreeing observations that are at
    least ``min_share`` of all concrete votes for that key; a later majority replaces
    an early wrong value. Colour keys may use ``color_min_share`` / ``color_min_votes``.
    For ``top_color``, a warm winner (red/pink) is rejected when cool colours
    collectively outvote it (neighbour red-tee contamination).
    """
    out = empty_features()
    color_share = float(min_share if color_min_share is None else color_min_share)
    color_need = int(min_votes if color_min_votes is None else color_min_votes)
    for key in FEATURE_KEYS:
        counter = votes.get(key)
        if not counter:
            if current is not None:
                out[key] = str(current.get(key, UNKNOWN))
            continue
        # Tan shirts oscillate beige/brown — treat brown as beige for top voting.
        if key == "top_color" and "beige" in counter and "brown" in counter:
            counter = Counter(counter)
            counter["beige"] += counter.pop("brown")

        total = sum(counter.values())
        need = color_need if key in COLOR_FEATURE_KEYS else int(min_votes)
        need_share = color_share if key in COLOR_FEATURE_KEYS else float(min_share)

        ranked = counter.most_common()
        winner, count = ranked[0]
        tied = len(ranked) > 1 and ranked[1][1] == count

        # Neighbour in a red tee: only override a *weak* warm lead (tie / near-tie
        # or warm share < 35%). A clear red plurality (true red tee) is kept.
        if key == "top_color" and winner in _WARM_TOP_COLORS:
            warm = sum(counter[c] for c in _WARM_TOP_COLORS if c in counter)
            cool = sum(counter[c] for c in _COOL_TOP_COLORS if c in counter)
            second = ranked[1][1] if len(ranked) > 1 else 0
            # Exact tie for first, or warm is a small minority of all colour votes.
            weak_warm = (count == second) or (warm / max(1, total) < 0.35)
            if weak_warm and cool >= warm:
                cool_only = Counter({c: n for c, n in counter.items() if c in _COOL_TOP_COLORS})
                if not cool_only:
                    out[key] = UNKNOWN
                    continue
                winner, count = cool_only.most_common(1)[0]
                tied = len(cool_only) > 1 and cool_only.most_common(2)[1][1] == count
                total = sum(cool_only.values())

        unique = count > (ranked[1][1] if len(ranked) > 1 else 0)
        share_ok = total > 0 and count / total >= need_share
        # Noisy CCTV colours rarely reach 55% — a unique plurality with enough
        # votes is still safer than leaving the gallery blank. Two clean agreeing
        # colour/pattern votes (unique) are enough to commit.
        plurality_ok = (
            key in COLOR_FEATURE_KEYS
            and unique
            and count >= max(need, 2)
            and not tied
        )
        if count >= max(1, need) and not tied and (share_ok or plurality_ok):
            out[key] = winner
        else:
            out[key] = UNKNOWN
    return out


def consensus_embedding(vectors: Sequence[np.ndarray]) -> np.ndarray | None:
    """Pick the embedding most similar to the others (majority-consistent sample)."""
    if not vectors:
        return None
    if len(vectors) == 1:
        return np.asarray(vectors[0], dtype=np.float32)
    m = np.stack([np.asarray(v, dtype=np.float32).reshape(-1) for v in vectors], axis=0)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    m = m / np.maximum(norms, 1e-8)
    sim = m @ m.T
    np.fill_diagonal(sim, 0.0)
    scores = sim.sum(axis=1)
    return m[int(np.argmax(scores))]


class VlmFeatureError(RuntimeError):
    pass


@dataclass
class VlmExtractResult:
    """Parsed features plus provenance for consensus / async matching."""
    features: dict[str, str]
    raw_by_key: dict[str, str]
    crop_sha1: str
    track_id: int | str | None = None
    frame_index: int | None = None
    timestamp_seconds: float | None = None
    head_shape: tuple[int, int] | None = None  # (w, h) of head crop used
    latency_ms: float = 0.0


def create_feature_extractor(config: VlmFeatureConfig | None = None) -> "MoondreamFeatureExtractor":
    """Pick Moondream or Qwen2.5-VL from ``config.model_id`` (same extract API)."""
    cfg = config or VlmFeatureConfig()
    mid = str(cfg.model_id or "").lower()
    if "qwen" in mid:
        # Moondream's pinned revision must not be applied to Qwen checkpoints.
        if str(cfg.revision or "").startswith("2025-"):
            cfg = replace(cfg, revision="")
        return QwenVlFeatureExtractor(cfg)
    return MoondreamFeatureExtractor(cfg)


class MoondreamFeatureExtractor:
    """Lazy Moondream2 wrapper — body + head single-attribute queries."""

    def __init__(self, config: VlmFeatureConfig | None = None) -> None:
        self.config = config or VlmFeatureConfig()
        self._model: Any = None
        self._device: str = "cpu"
        self._dtype: Any = None
        self._lock = threading.Lock()
        self._extract_seq = 0

    def _resolve_device(self) -> str:
        if self.config.device != "auto":
            return self.config.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                import torch
                from transformers import AutoModelForCausalLM
            except ImportError as exc:
                raise VlmFeatureError("transformers/torch required for VLM features") from exc
            device = self._resolve_device()
            logger.info(
                "Loading Moondream2 for structured features (%s @ %s) on %s (native bfloat16)",
                self.config.model_id, self.config.revision, device,
            )
            kwargs: dict[str, Any] = {
                "trust_remote_code": True,
            }
            if self.config.revision:
                kwargs["revision"] = self.config.revision
            if device.startswith("cuda"):
                kwargs["device_map"] = {"": device}
            model = AutoModelForCausalLM.from_pretrained(self.config.model_id, **kwargs)
            if not device.startswith("cuda"):
                model = model.to(device)
            model.eval()
            self._model = model
            self._device = device
            self._dtype = torch.bfloat16
            logger.info("Moondream2 feature extractor ready (body+head single-attr queries)")
            return self._model

    @staticmethod
    def _to_pil(bgr_crop: np.ndarray) -> Image.Image:
        if bgr_crop is None or bgr_crop.size == 0:
            raise VlmFeatureError("empty crop")
        return Image.fromarray(cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB))

    def _query(self, model: Any, image: Image.Image, prompt: str) -> str:
        out = model.query(image, prompt)
        return str(out.get("answer") or out.get("caption") or "").strip()

    def _debug_dump(
        self,
        bgr_crop: np.ndarray,
        result: VlmExtractResult,
        head_crop: np.ndarray | None = None,
        torso_crop: np.ndarray | None = None,
        lower_crop: np.ndarray | None = None,
    ) -> None:
        root = self.config.debug_directory
        if root is None:
            return
        try:
            root = Path(root)
            root.mkdir(parents=True, exist_ok=True)
            with self._lock:
                self._extract_seq += 1
                seq = self._extract_seq
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            tid = "na" if result.track_id is None else str(result.track_id)
            fr = "na" if result.frame_index is None else str(result.frame_index)
            stem = f"{stamp}_seq{seq:05d}_T{tid}_f{fr}"
            cv2.imwrite(str(root / f"{stem}.jpg"), bgr_crop)
            if head_crop is not None and head_crop.size:
                cv2.imwrite(str(root / f"{stem}_head.jpg"), head_crop)
            if torso_crop is not None and torso_crop.size:
                cv2.imwrite(str(root / f"{stem}_torso.jpg"), torso_crop)
            if lower_crop is not None and lower_crop.size:
                cv2.imwrite(str(root / f"{stem}_lower.jpg"), lower_crop)
            raw_block = "\n".join(f"{k}: {result.raw_by_key.get(k, '')}" for k in FEATURE_KEYS)
            meta = (
                f"track_id={result.track_id}\n"
                f"frame_index={result.frame_index}\n"
                f"timestamp_seconds={result.timestamp_seconds}\n"
                f"crop_sha1={result.crop_sha1}\n"
                f"head_shape={result.head_shape}\n"
                f"latency_ms={result.latency_ms:.1f}\n"
                f"saved_utc={datetime.now(timezone.utc).isoformat()}\n"
                f"\n--- raw_by_key ---\n{raw_block}\n"
                f"\n--- parsed ---\n"
                + "\n".join(f"{k}: {result.features.get(k, UNKNOWN)}" for k in FEATURE_KEYS)
                + "\n"
            )
            (root / f"{stem}.txt").write_text(meta, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("VLM debug dump failed: %s", exc)

    def extract(
        self,
        bgr_crop: np.ndarray,
        *,
        track_id: int | str | None = None,
        frame_index: int | None = None,
        timestamp_seconds: float | None = None,
        expect_sha1: str | None = None,
    ) -> dict[str, str]:
        """One crop → feature dict (backward-compatible for ThreadPool callers)."""
        result = self.extract_detailed(
            bgr_crop,
            track_id=track_id,
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            expect_sha1=expect_sha1,
        )
        return result.features

    def extract_detailed(
        self,
        bgr_crop: np.ndarray,
        *,
        track_id: int | str | None = None,
        frame_index: int | None = None,
        timestamp_seconds: float | None = None,
        expect_sha1: str | None = None,
    ) -> VlmExtractResult:
        """Gender on full crop; colours on torso/lower; accessories on head (no upscale)."""
        import torch

        sha = crop_sha1(bgr_crop)
        if expect_sha1 is not None and expect_sha1 != sha:
            raise VlmFeatureError(
                f"crop hash mismatch: expected {expect_sha1[:12]} got {sha[:12]} "
                f"(async result does not match submitted crop)"
            )

        model = self._ensure_loaded()
        cfg = self.config
        body_image = self._to_pil(bgr_crop)
        head = head_crop_from_person(
            bgr_crop,
            height_fraction=cfg.head_height_fraction,
            pad_x_fraction=cfg.head_pad_x_fraction,
        )
        head_ok = head_crop_usable(head, min_side_px=cfg.min_head_side_px)
        head_image = self._to_pil(head) if head_ok and head is not None else None
        head_shape = (int(head.shape[1]), int(head.shape[0])) if head is not None else None

        torso = torso_crop_from_person(
            bgr_crop,
            y0_fraction=cfg.torso_y0_fraction,
            y1_fraction=cfg.torso_y1_fraction,
        )
        lower = lower_crop_from_person(
            bgr_crop,
            y0_fraction=cfg.lower_y0_fraction,
            y1_fraction=cfg.lower_y1_fraction,
        )
        torso_usable = clothing_crop_usable(
            torso, min_side_px=cfg.min_color_side_px,
            min_mean=cfg.min_color_mean, max_mean=cfg.max_color_mean,
        )
        lower_usable = clothing_crop_usable(
            lower, min_side_px=cfg.min_color_side_px,
            min_mean=cfg.min_color_mean, max_mean=cfg.max_color_mean,
        )
        torso_q = clothing_crop_quality(torso) if torso_usable else 0.0
        lower_q = clothing_crop_quality(lower) if lower_usable else 0.0
        # Only mid-quality+ crops vote colours/patterns (matches human clear-frame review).
        min_q = float(getattr(cfg, "min_color_quality", 0.35) or 0.35)
        color_ok = {
            "top_color": torso_usable and torso_q >= min_q,
            "bottom_color": lower_usable and lower_q >= min_q,
            "top_pattern": torso_usable and torso_q >= min_q,
        }
        torso_image = self._to_pil(torso) if torso_usable and torso is not None else None
        lower_image = self._to_pil(lower) if lower_usable and lower is not None else None
        color_images = {
            "top_color": torso_image if color_ok["top_color"] else None,
            "bottom_color": lower_image if color_ok["bottom_color"] else None,
            "top_pattern": torso_image if color_ok["top_pattern"] else None,
        }

        raw_by_key: dict[str, str] = {}
        features = empty_features()
        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                for key in BODY_ATTR_KEYS:
                    text = self._query(model, body_image, SINGLE_ATTR_PROMPTS[key])
                    raw_by_key[key] = text
                    features[key] = parse_single_attr(key, text)
                for key in (*COLOR_ATTR_KEYS, *TORSO_ATTR_KEYS):
                    img = color_images.get(key)
                    if img is None:
                        raw_by_key[key] = ""
                        features[key] = UNKNOWN
                    else:
                        text = self._query(model, img, SINGLE_ATTR_PROMPTS[key])
                        raw_by_key[key] = text
                        features[key] = parse_single_attr(key, text)
                    # Torso/lower often too dark: if answer is black/unknown/gray,
                    # re-ask on the full person crop and keep a chromatic reply.
                    # Pattern: retry full crop when torso said unknown.
                    retry_weak = (
                        features[key] in {UNKNOWN, "black", "gray"}
                        if key in COLOR_ATTR_KEYS
                        else features[key] == UNKNOWN
                    )
                    if retry_weak:
                        text2 = self._query(model, body_image, SINGLE_ATTR_PROMPTS[key])
                        parsed2 = parse_single_attr(key, text2)
                        raw_by_key[key] = f"{raw_by_key.get(key, '')}|full:{text2}".strip("|")
                        if key in COLOR_ATTR_KEYS:
                            if parsed2 not in {UNKNOWN, "black", "gray"}:
                                features[key] = parsed2
                            elif features[key] == UNKNOWN and parsed2 != UNKNOWN:
                                features[key] = parsed2
                        elif features[key] == UNKNOWN and parsed2 != UNKNOWN:
                            features[key] = parsed2
                for key in HEAD_ATTR_KEYS:
                    if not head_ok or head_image is None:
                        raw_by_key[key] = ""
                        features[key] = UNKNOWN
                        continue
                    text = self._query(model, head_image, SINGLE_ATTR_PROMPTS[key])
                    raw_by_key[key] = text
                    features[key] = parse_single_attr(key, text)
        except VlmFeatureError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VlmFeatureError(f"feature extract failed: {exc}") from exc

        # CV assist: weak VLM blacks → green/blue; sanitize false beige/blue.
        if color_ok.get("top_color"):
            cv_top = estimate_clothing_color_cv(torso)
            features["top_color"] = refine_color_with_cv(features["top_color"], cv_top)
            features["top_color"] = sanitize_top_color(features["top_color"], torso)
        if color_ok.get("bottom_color"):
            cv_bot = estimate_clothing_color_cv(lower)
            features["bottom_color"] = refine_color_with_cv(features["bottom_color"], cv_bot)
            features["bottom_color"] = sanitize_bottom_color(features["bottom_color"], lower)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        result = VlmExtractResult(
            features=features,
            raw_by_key=raw_by_key,
            crop_sha1=sha,
            track_id=track_id,
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            head_shape=head_shape,
            latency_ms=elapsed_ms,
        )
        logger.info(
            "VLM features T%s frame=%s t=%.2fs sha=%s head=%s (%.0fms): raw=%s parsed=%s",
            track_id if track_id is not None else "?",
            frame_index if frame_index is not None else "?",
            float(timestamp_seconds) if timestamp_seconds is not None else -1.0,
            sha[:12],
            head_shape,
            elapsed_ms,
            {k: raw_by_key.get(k, "")[:40] for k in FEATURE_KEYS},
            features,
        )
        self._debug_dump(
            bgr_crop, result, head_crop=head, torso_crop=torso, lower_crop=lower,
        )
        return result

    # Alias used by older call sites / tests
    def extract_with_raw(
        self,
        bgr_crop: np.ndarray,
        *,
        track_id: int | str | None = None,
        frame_index: int | None = None,
        timestamp_seconds: float | None = None,
    ) -> tuple[dict[str, str], str]:
        result = self.extract_detailed(
            bgr_crop,
            track_id=track_id,
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
        )
        raw_joined = "\n".join(f"{k}: {result.raw_by_key.get(k, '')}" for k in FEATURE_KEYS)
        return result.features, raw_joined


class QwenVlFeatureExtractor(MoondreamFeatureExtractor):
    """Qwen2.5-VL backend — same crop/parse pipeline, stronger CCTV colour answers.

    Uses the official chat+vision path (``qwen_vl_utils``). Inference is serialized
    under ``_lock`` so hybrid's thread-pool workers do not collide on one GPU.
    Prefer ``Qwen/Qwen2.5-VL-3B-Instruct`` on ~12GB cards (fits with YOLO+ReID).
    """

    def __init__(self, config: VlmFeatureConfig | None = None) -> None:
        super().__init__(config)
        self._processor: Any = None

    def _ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                import torch
                from transformers import AutoProcessor
            except ImportError as exc:
                raise VlmFeatureError("transformers/torch required for VLM features") from exc
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLModel
            except ImportError:
                try:
                    from transformers import Qwen2VLForConditionalGeneration as QwenVLModel
                except ImportError as exc:
                    raise VlmFeatureError(
                        "transformers build lacks Qwen2.5-VL; upgrade transformers>=4.49"
                    ) from exc
            try:
                from qwen_vl_utils import process_vision_info  # noqa: F401
            except ImportError as exc:
                raise VlmFeatureError(
                    "qwen-vl-utils is required for Qwen2.5-VL features "
                    "(pip install qwen-vl-utils)"
                ) from exc

            device = self._resolve_device()
            rev = str(self.config.revision or "").strip()
            logger.info(
                "Loading Qwen2.5-VL for structured features (%s%s) on %s (bfloat16)",
                self.config.model_id,
                f" @ {rev}" if rev else "",
                device,
            )
            dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
            load_kwargs: dict[str, Any] = {
                "torch_dtype": dtype,
                "trust_remote_code": True,
            }
            if rev:
                load_kwargs["revision"] = rev
            if device.startswith("cuda"):
                load_kwargs["device_map"] = {"": device}
            model = QwenVLModel.from_pretrained(self.config.model_id, **load_kwargs)
            if not device.startswith("cuda"):
                model = model.to(device)
            model.eval()
            proc_kwargs: dict[str, Any] = {"trust_remote_code": True}
            if rev:
                proc_kwargs["revision"] = rev
            processor = AutoProcessor.from_pretrained(self.config.model_id, **proc_kwargs)
            self._model = model
            self._processor = processor
            self._device = device
            self._dtype = dtype
            logger.info("Qwen2.5-VL feature extractor ready (body+head single-attr queries)")
            return self._model

    def _query(self, model: Any, image: Image.Image, prompt: str) -> str:
        """One image + short prompt → answer text (GPU generate under lock)."""
        from qwen_vl_utils import process_vision_info

        processor = self._processor
        if processor is None:
            raise VlmFeatureError("Qwen processor not loaded")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        # Serialize generate — hybrid submits from a thread pool.
        # The GPU gate keeps this from overlapping the detector.
        from core.gpu_gate import gpu_gate

        with gpu_gate(), self._lock:
            kwargs: dict[str, Any] = {
                "text": [text],
                "images": image_inputs,
                "padding": True,
                "return_tensors": "pt",
            }
            if video_inputs is not None:
                kwargs["videos"] = video_inputs
            inputs = processor(**kwargs)
            inputs = inputs.to(self._device)
            max_new = max(8, int(self.config.max_new_tokens or 24))
            generated = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
            trimmed = [
                out[len(inp):] for inp, out in zip(inputs.input_ids, generated)
            ]
            answer = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )[0]
        return str(answer or "").strip()
