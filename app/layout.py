"""Shared Streamlit chrome. No model code lives here."""

from __future__ import annotations

import bootstrap  # noqa: F401
from pathlib import Path

import streamlit as st

from pipeline.config import ROOT_DIR


def setup(page_title: str):
    st.set_page_config(
        page_title=f"{page_title} · Workplace CV",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _css()
    cfg = _load_settings()
    _sidebar(cfg)
    return cfg


def _load_settings():
    from pipeline.config import load_settings

    return load_settings()


def _css() -> None:
    st.markdown(
        """
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px;}
            h1, h2, h3 {letter-spacing: -0.02em;}
            div[data-testid="stMetric"] {
                background: #171d27;
                border: 1px solid #243044;
                padding: 12px 14px;
                border-radius: 10px;
            }
            div[data-testid="stSidebar"] {background: #121821;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sidebar(settings) -> None:
    yolo = settings.resolved_yolo_path()
    custom = Path(settings.yolo_model_path)
    if not custom.is_absolute():
        custom = ROOT_DIR / custom
    yolo_note = "custom" if custom.is_file() else "fallback yolov8n"
    extra = len(settings.extra_occupancy_sources)
    from pipeline.capture import mask_source
    with st.sidebar:
        st.markdown("**Workplace CV**")
        st.caption("Face attendance · desk occupancy")
        st.divider()
        st.caption("SOURCE")
        st.write(settings.source_mode.upper())
        st.caption("FACE CAMERA")
        st.write(mask_source(settings.face_source))
        st.caption("OCCUPANCY CAMERA")
        st.write(mask_source(settings.occupancy_source))
        if extra:
            st.caption(f"{extra} extra occupancy source(s) in .env")
        st.divider()
        st.caption("MODELS")
        st.write(f"Face  ·  {settings.insightface_model}")
        st.write(f"YOLO  ·  {yolo_note}")
        st.caption(yolo)


@st.cache_resource
def get_store():
    from pipeline.store import Store

    return Store()


@st.cache_resource
def get_face_engine():
    from pipeline.face_engine import FaceEngine

    return FaceEngine()
