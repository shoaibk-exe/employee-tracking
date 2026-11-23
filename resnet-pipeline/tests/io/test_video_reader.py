import time

import cv2
import numpy as np
import pytest

from core.io.video_reader import FramePacket, ReconnectPolicy, SourceOutage, VideoReader, VideoSourceError


def write_video(path, frames=30, fps=10, size=(64, 48)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(frames):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        frame[:, : size[0] // 2] = (i * 8) % 256
        writer.write(frame)
    writer.release()
    return path


def test_file_timestamps_and_sampling(tmp_path):
    path = write_video(tmp_path / "v.mp4")
    with VideoReader(str(path), inference_interval_seconds=1.0) as reader:
        assert (reader.width, reader.height) == (64, 48) and reader.fps == pytest.approx(10)
        packets = list(reader)
    assert [round(p.timestamp_seconds, 2) for p in packets] == [0.0, 1.0, 2.0]
    assert all(p.timestamp_source == "pts" and p.run_inference for p in packets)
    assert [p.frame_index for p in packets] == [0, 10, 20]


def test_yield_all_frames_flags_inference(tmp_path):
    path = write_video(tmp_path / "v.mp4", frames=15)
    packets = list(VideoReader(str(path), 0.5, yield_all_frames=True))
    assert len(packets) == 15
    assert [p.frame_index for p in packets if p.run_inference] == [0, 5, 10]


def test_rotation_changes_dimensions(tmp_path):
    path = write_video(tmp_path / "v.mp4", frames=3)
    reader = VideoReader(str(path), 0, rotation=90)
    assert (reader.width, reader.height) == (48, 64)
    assert next(iter(reader)).frame.shape[:2] == (64, 48)
    reader.close()


def test_missing_file():
    with pytest.raises(VideoSourceError):
        VideoReader("does/not/exist.mp4")


class FakeLiveCapture:
    def __init__(self, frames_before_failure, fail_forever=False):
        self.remaining = frames_before_failure
        self.fail_forever = fail_forever
        self.opened = True

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH: 32, cv2.CAP_PROP_FRAME_HEIGHT: 24, cv2.CAP_PROP_FPS: 50}.get(prop, 0)

    def read(self):
        time.sleep(0.02)
        if self.remaining > 0:
            self.remaining -= 1
            return True, np.zeros((24, 32, 3), dtype=np.uint8)
        return False, None

    def release(self):
        self.opened = False


def test_live_outage_and_reconnect():
    captures = [FakeLiveCapture(10), FakeLiveCapture(10)]

    def factory(source, policy):
        return captures.pop(0) if captures else FakeLiveCapture(0)

    policy = ReconnectPolicy(read_failure_tolerance=2, reconnect_delay_seconds=0.05,
                             max_reconnect_delay_seconds=0.05, max_reconnect_attempts=1, read_timeout_seconds=0.5)
    reader = VideoReader("rtsp://fake/stream", 0, reconnect=policy, capture_factory=factory)
    items = list(reader)
    reader.close()
    outages = [i for i in items if isinstance(i, SourceOutage)]
    packets = [i for i in items if isinstance(i, FramePacket)]
    assert len(outages) == 2
    assert packets and all(p.timestamp_source == "monotonic" and p.wall_clock_utc for p in packets)
    assert sum(p.after_reconnect for p in packets) == 1
    ts = [p.timestamp_seconds for p in packets]
    assert ts == sorted(ts)
    assert reader.source_lost
