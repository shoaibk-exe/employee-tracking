"""Credential redaction for RTSP error messages."""

from api.redact import redact_rtsp


def test_redact_rtsp_strips_userinfo():
    assert (
        redact_rtsp("rtsp://admin:secret@10.0.0.5:554/stream1")
        == "rtsp://10.0.0.5:554/stream1"
    )


def test_redact_rtsp_leaves_plain_urls():
    assert redact_rtsp("rtsp://10.0.0.5/live") == "rtsp://10.0.0.5/live"
