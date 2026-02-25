# logging/observers.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .base import BaseObserver
from .events import Event
from .jsonl import JsonlWriter


@dataclass
class JsonlObserver(BaseObserver):
    w: JsonlWriter = None  # type: ignore


@dataclass
class StepLogger(JsonlObserver):
    every_s: Optional[float] = 0.5
    track_id: Optional[int] = None

    def set_track_id(self, agent_id: int) -> None:
        self.track_id = int(agent_id)

    def handle(self, ev: Event) -> None:
        if ev.name != "step":
            return
        if not self.allow(ev.t):
            return
        if self.track_id is not None:
            sid = int(ev.payload.get("summary", {}).get("agent_id", -1))
            if sid != int(self.track_id):
                return
        self.w.write(ev.payload)

@dataclass
class SampleLogger(JsonlObserver):
    every_s: Optional[float] = 1.0  # t.ex. en gång per sekund

    def handle(self, ev: Event) -> None:
        if ev.name != "sample":
            return
        if not self.allow(ev.t):
            return
        self.w.write(ev.payload)
        
@dataclass
class PopLogger(JsonlObserver):
    every_s: Optional[float] = 1.0

    def handle(self, ev: Event) -> None:
        if ev.name != "population":
            return
        if not self.allow(ev.t):
            return
        self.w.write(ev.payload)


@dataclass
class LifeLogger(JsonlObserver):
    every_s: Optional[float] = None  # no gating by default

    def handle(self, ev: Event) -> None:
        if ev.name not in ("birth", "death"):
            return
        if not self.allow(ev.t):
            return
        self.w.write(ev.payload)


@dataclass
class WorldLogger(JsonlObserver):
    every_s: Optional[float] = 2.0

    def handle(self, ev: Event) -> None:
        if ev.name != "world":
            return
        if not self.allow(ev.t):
            return
        self.w.write(ev.payload)