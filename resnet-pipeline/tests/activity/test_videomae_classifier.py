"""Unit tests for VideoMAE activity helpers (no GPU model load)."""

from __future__ import annotations

import unittest

import numpy as np

from core.activity.videomae_classifier import _square_resize, clip_span, sample_indices


class VideoMAEHelperTests(unittest.TestCase):
    def test_clip_span_matches_training(self) -> None:
        self.assertEqual(clip_span(16, 4), 61)

    def test_sample_indices_full_buffer(self) -> None:
        indices = sample_indices(80, 16, 4)
        self.assertEqual(len(indices), 16)
        self.assertEqual(indices[0], 80 - 61)
        self.assertEqual(indices[-1], 79)
        self.assertEqual(indices[1] - indices[0], 4)

    def test_sample_indices_short_buffer(self) -> None:
        indices = sample_indices(20, 16, 4)
        self.assertEqual(len(indices), 16)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 19)

    def test_square_resize_center_crop(self) -> None:
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        frame[:, 50:150] = 255
        out = _square_resize(frame, 64)
        self.assertEqual(out.shape, (64, 64, 3))
        self.assertGreater(int(out.mean()), 200)


if __name__ == "__main__":
    unittest.main()
