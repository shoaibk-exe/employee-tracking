"""Local VLM (Moondream2) person-appearance describer for same-day P## matching.

Uses clothing / visible attributes only — not faces — and is wiped with the
day gallery. This is anonymous appearance labeling, not verified identity.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

DESCRIBE_PROMPT = (
    "Describe only this person's visible clothing and appearance for re-identification. "
    "Include: top clothing type and color, bottom clothing type and color, "
    "outerwear if any, shoes if visible, hair length/color if visible, "
    "and any distinctive accessory (bag, hat, scarf). "
    "Ignore face, pose, and background. One short sentence."
)

MATCH_PROMPT_TEMPLATE = (
    "Compare the person in the image to these candidates. Match ONLY if the "
    "top clothing color clearly matches. If top colors differ, answer NEW.\n"
    "{candidates}\n"
    "Reply with ONLY one token: a candidate ID like P01, or NEW."
)

_COLORS = (
    "navy", "black", "white", "gray", "grey", "blue", "red", "pink", "green",
    "yellow", "orange", "brown", "beige", "purple", "maroon", "khaki", "denim",
    "teal", "cyan", "cream", "tan",
)
# Collapse near-synonyms so "navy top" rematches "blue shirt".
_COLOR_CANON = {
    "navy": "blue",
    "denim": "blue",
    "grey": "gray",
    "teal": "green",
    "cyan": "blue",
    "cream": "white",
    "tan": "beige",
    "maroon": "red",
}
_TOP_WORDS = (
    "shirt", "t-shirt", "tee", "blouse", "hoodie", "sweater", "jacket", "coat",
    "vest", "top", "pullover", "cardigan", "outerwear",
)
_BOTTOM_WORDS = (
    "pants", "jeans", "trousers", "shorts", "skirt", "leggings", "bottoms", "bottom",
)
_ACCESSORY_WORDS = ("bag", "backpack", "hat", "cap", "scarf", "glasses", "helmet", "shoes", "sneakers")


class VlmDescriberError(RuntimeError):
    """Raised when the VLM cannot load or answer."""


@dataclass
class VlmDescriberConfig:
    model_id: str = "vikhyatk/moondream2"
    revision: str = "2025-06-21"
    device: str = "auto"
    max_new_tokens: int = 64
    # Description-token match: accept if score >= this and beats 2nd by margin.
    match_min_score: float = 2.5
    match_margin: float = 1.0


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _canon_color(color: str) -> str:
    return _COLOR_CANON.get(color, color)


def _colors_in_text_order(text: str) -> list[str]:
    """Return canonical color words in the order they appear in ``text``."""
    hits: list[tuple[int, str]] = []
    for c in _COLORS:
        for m in re.finditer(rf"\b{re.escape(c)}\b", text):
            hits.append((m.start(), _canon_color(c)))
    hits.sort(key=lambda x: x[0])
    # Dedupe while preserving order.
    out: list[str] = []
    seen: set[str] = set()
    for _, c in hits:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def extract_appearance_tokens(text: str) -> dict[str, set[str]]:
    """Pull coarse clothing tokens from a VLM description sentence."""
    t = _normalize(text)
    ordered_all = _colors_in_text_order(t)
    colors = set(ordered_all)
    tops = {w for w in _TOP_WORDS if re.search(rf"\b{re.escape(w)}\b", t)}
    bottoms = {w for w in _BOTTOM_WORDS if re.search(rf"\b{re.escape(w)}\b", t)}
    accessories = {w for w in _ACCESSORY_WORDS if re.search(rf"\b{re.escape(w)}\b", t)}

    # Heuristic: colors mentioned near top/bottom garment words (text-order-aware).
    top_colors: set[str] = set()
    bottom_colors: set[str] = set()
    parts = re.split(r",| and |;|\.", t)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        has_top = any(re.search(rf"\b{re.escape(w)}\b", part) for w in _TOP_WORDS)
        has_bottom = any(re.search(rf"\b{re.escape(w)}\b", part) for w in _BOTTOM_WORDS)
        ordered = _colors_in_text_order(part)
        if has_top and has_bottom and ordered:
            top_colors.add(ordered[0])
            bottom_colors.add(ordered[-1] if len(ordered) > 1 else ordered[0])
        elif has_top:
            top_colors.update(ordered)
        elif has_bottom:
            bottom_colors.update(ordered)
    if not top_colors and ordered_all:
        top_colors.add(ordered_all[0])
        if len(ordered_all) > 1 and not bottom_colors:
            bottom_colors.add(ordered_all[1])

    return {
        "colors": colors,
        "top_colors": top_colors,
        "bottom_colors": bottom_colors,
        "tops": tops,
        "bottoms": bottoms,
        "accessories": accessories,
    }


def appearance_similarity(a: str, b: str) -> float:
    """Weighted overlap of clothing attributes (higher = more similar)."""
    ta, tb = extract_appearance_tokens(a), extract_appearance_tokens(b)
    score = 0.0
    top_overlap = ta["top_colors"] & tb["top_colors"]
    bottom_overlap = ta["bottom_colors"] & tb["bottom_colors"]
    if top_overlap:
        score += 3.0
    elif ta["top_colors"] and tb["top_colors"]:
        # Explicit conflicting top colors → hard penalty.
        score -= 3.0
    color_overlap = ta["colors"] & tb["colors"]
    score += 0.5 * len(color_overlap)
    if bottom_overlap:
        score += 2.0
    score += 0.75 * len(ta["tops"] & tb["tops"])
    score += 0.75 * len(ta["bottoms"] & tb["bottoms"])
    score += 0.5 * len(ta["accessories"] & tb["accessories"])
    return score


def best_description_match(
    query_desc: str,
    candidates: dict[str, str],
    *,
    min_score: float = 2.5,
    margin: float = 1.0,
) -> tuple[str | None, float, float]:
    """Return (person_id|None, best_score, second_score) from description overlap."""
    if not candidates:
        return None, -1.0, -1.0
    scored = [(pid, appearance_similarity(query_desc, desc)) for pid, desc in candidates.items()]
    scored.sort(key=lambda x: -x[1])
    best_id, best = scored[0]
    second = scored[1][1] if len(scored) > 1 else -1.0
    if best < min_score:
        return None, best, second
    if second >= 0.0 and (best - second) < margin:
        return None, best, second
    # Require non-conflicting top colors when both have them.
    tq = extract_appearance_tokens(query_desc)
    tc = extract_appearance_tokens(candidates[best_id])
    if tq["top_colors"] and tc["top_colors"] and not (tq["top_colors"] & tc["top_colors"]):
        return None, best, second
    return best_id, best, second


class MoondreamDescriber:
    """Lazy-loaded Moondream2 query wrapper (thread-safe single load)."""

    def __init__(self, config: VlmDescriberConfig | None = None) -> None:
        self.config = config or VlmDescriberConfig()
        self._model: Any = None
        self._device: str = "cpu"
        self._lock = threading.Lock()

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
                raise VlmDescriberError(
                    "transformers/torch required for VLM appearance gallery"
                ) from exc

            device = self._resolve_device()
            # Native bfloat16 only — never torch_dtype / model.to(dtype=...); casting
            # breaks Moondream's hardcoded bfloat16 vision path.
            logger.info(
                "Loading Moondream2 VLM (%s @ %s) on %s (native bfloat16) — first download may take a few minutes",
                self.config.model_id, self.config.revision, device,
            )
            # Moondream remote code needs transformers~=4.49 (5.x breaks / garbles output).
            kwargs: dict[str, Any] = {
                "revision": self.config.revision,
                "trust_remote_code": True,
            }
            if device.startswith("cuda"):
                kwargs["device_map"] = {"": device}

            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id, **kwargs,
            )
            if not device.startswith("cuda"):
                model = model.to(device)
            model.eval()
            self._model = model
            self._device = device
            logger.info("Moondream2 VLM ready")
            return self._model

    @staticmethod
    def _to_pil(bgr_crop: np.ndarray) -> Image.Image:
        if bgr_crop is None or bgr_crop.size == 0:
            raise VlmDescriberError("empty crop")
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def describe(self, bgr_crop: np.ndarray) -> str:
        """Return a short clothing/appearance sentence for the person crop."""
        import torch

        model = self._ensure_loaded()
        image = self._to_pil(bgr_crop)
        try:
            with torch.inference_mode():
                out = model.query(image, DESCRIBE_PROMPT)
            text = str(out.get("answer") or out.get("caption") or "").strip()
        except Exception as exc:  # noqa: BLE001 — surface as gallery soft-fail
            raise VlmDescriberError(f"describe failed: {exc}") from exc
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            raise VlmDescriberError("empty VLM description")
        return text

    def match_id(
        self,
        bgr_crop: np.ndarray,
        candidates: dict[str, str],
        *,
        query_description: str | None = None,
    ) -> str | None:
        """Match crop to a free gallery description, or None for NEW.

        Primary: clothing-token similarity on a fresh VLM description.
        Fallback: ask the VLM to pick an ID when token scores are inconclusive
        but only one candidate exists.
        """
        if not candidates:
            return None

        desc = query_description
        if not desc:
            desc = self.describe(bgr_crop)

        best_id, best, second = best_description_match(
            desc,
            candidates,
            min_score=float(self.config.match_min_score),
            margin=float(self.config.match_margin),
        )
        if best_id is not None:
            logger.debug(
                "VLM desc-match %s score=%.2f (2nd=%.2f) desc=%r",
                best_id, best, second, desc[:80],
            )
            return best_id

        # Conflicting top colors / very weak overlap → always NEW (do not trust VLM pick).
        if best < 1.0:
            return None

        # Near-threshold ambiguity with 1–2 candidates → ask VLM, then re-check tokens.
        if len(candidates) > 2:
            return None

        model = self._ensure_loaded()
        image = self._to_pil(bgr_crop)
        lines = "\n".join(f"{pid}: {cdesc}" for pid, cdesc in candidates.items())
        prompt = MATCH_PROMPT_TEMPLATE.format(candidates=lines)
        try:
            out = model.query(image, prompt)
            raw = str(out.get("answer") or "").strip().upper()
        except Exception as exc:  # noqa: BLE001
            raise VlmDescriberError(f"match failed: {exc}") from exc

        raw = re.sub(r"\s+", " ", raw)
        if re.search(r"\bNEW\b", raw):
            return None
        found = re.findall(r"\bP\d{1,3}\b", raw)
        for token in found:
            pid = token if token in candidates else None
            if pid is None:
                try:
                    n = int(token.lstrip("P"))
                    padded = f"P{n:02d}"
                    if padded in candidates:
                        pid = padded
                except ValueError:
                    pid = None
            if pid is None:
                continue
            guard = appearance_similarity(desc, candidates[pid])
            if guard < float(self.config.match_min_score) * 0.6:
                return None
            return pid
        return None
