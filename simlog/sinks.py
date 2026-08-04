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

    def wants(self, name: str, t: float) -> bool:
        """
        Finns det någon observatör som skulle skriva det här eventet nu?

        Producenten kan fråga innan den bygger nyttolasten. Världsposten kostar
        en full florasammanfattning med kvantiler plus två svep över cell- och
        floravektorerna — vid 256x256 var det en fjärdedel av takten, byggd
        varje tick och kastad, eftersom hela avgörandet låg i `handle()`.
        """
        return any(ob.wants(name, float(t)) for ob in self.observers)
