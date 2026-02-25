# logging/events.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

EventName = Literal["step", "population", "birth", "death", "world", "sample"]

@dataclass(frozen=True)
class Event:
    name: EventName
    t: float
    payload: Dict[str, Any]