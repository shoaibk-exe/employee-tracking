from __future__ import annotations

import colorsys
import hashlib


def color_for_global_id(global_id: str) -> tuple[int, int, int]:
    """Stable BGR color for a global identity label."""
    digest = hashlib.md5(global_id.encode("utf-8")).hexdigest()
    hue = int(digest[:8], 16) % 360 / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def provisional_color() -> tuple[int, int, int]:
    return (0, 200, 255)  # amber-ish BGR


def unknown_color() -> tuple[int, int, int]:
    return (160, 160, 160)
