"""GPU and process health sampling."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GpuStats:
    memory_mb: float | None = None
    util_percent: float | None = None


class HealthMonitor:
    def __init__(self) -> None:
        self.started_at = time.time()
        self._last_gpu = GpuStats()

    def uptime_seconds(self) -> float:
        return time.time() - self.started_at

    def sample_gpu(self) -> GpuStats:
        try:
            import torch

            if not torch.cuda.is_available():
                self._last_gpu = GpuStats()
                return self._last_gpu
            mem = torch.cuda.memory_allocated(0) / (1024 ** 2)
            # Utilization requires NVML; best-effort.
            util = None
            try:
                import pynvml  # type: ignore

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            except Exception:  # noqa: BLE001
                util = None
            self._last_gpu = GpuStats(memory_mb=float(mem), util_percent=util)
        except Exception as exc:  # noqa: BLE001
            logger.debug("GPU sample failed: %s", exc)
            self._last_gpu = GpuStats()
        return self._last_gpu

    @property
    def last_gpu(self) -> GpuStats:
        return self._last_gpu
