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

    # Eventnamn den här observatören alls bryr sig om. Tom betyder alla, för
    # bakåtkompatibilitet med observatörer som filtrerar själva i handle().
    event_names: tuple = ()

    def wants(self, name: str, t: float) -> bool:
        """
        Skulle `handle()` skriva något för det här eventet vid den här tiden?

        Sidoeffektfri, till skillnad från `allow()`, som flyttar fram nästa
        tillåtna tidpunkt. Skillnaden är hela poängen: producenten frågar först
        och bygger nyttolasten bara om svaret är ja, medan `allow()` fortfarande
        är den som avgör när posten väl kommer.
        """
        if self.event_names and name not in self.event_names:
            return False
        return self.every_s is None or float(t) >= self._next_t

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