"""Live capture / engine handles that Streamlit cannot pickle into session_state."""

from __future__ import annotations

from typing import Any, Callable

captures: dict[str, Any] = {}
engines: dict[str, Any] = {}


def get_capture(name: str, factory: Callable[[], Any], source: str = "") -> Any:
    cap = captures.get(name)
    if cap is not None and source:
        current = getattr(cap, "source", "")
        if current != source:
            release_capture(name)
            cap = None
    if cap is None:
        cap = factory()
        captures[name] = cap
    return cap


def release_capture(name: str) -> None:
    cap = captures.pop(name, None)
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass


def get_engine(name: str, factory: Callable[[], Any]) -> Any:
    eng = engines.get(name)
    if eng is None:
        eng = factory()
        engines[name] = eng
    return eng
