# logging/sinks.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List

from .events import Event
from .base import BaseObserver

@dataclass
class EventHub:
    observers: List[BaseObserver]

    def emit(self, ev: Event) -> None:
        for ob in self.observers:
            ob.handle(ev)