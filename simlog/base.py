# logging/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .events import Event


@dataclass
class BaseObserver:
    """
    Base class with optional sim-time gating (every_s).
    If every_s is None: no gating.
    """
    every_s: Optional[float] = None
    _next_t: float = 0.0

    def allow(self, t: float) -> bool:
        if self.every_s is None:
            return True
        tt = float(t)
        if tt < self._next_t:
            return False
        self._next_t = tt + float(self.every_s)
        return True

    def handle(self, ev: Event) -> None:
        raise NotImplementedError