# jsonl.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, TextIO


@dataclass
class JsonlWriter:
    fp: str
    flush_every: int = 1  # 1 = flush varje rad (bra för tail -f), höj till 10/50 för mindre overhead
    _f: Optional[TextIO] = None
    _n: int = 0

    def __enter__(self) -> "JsonlWriter":
        # line-buffered (buffering=1) hjälper vid tail -f
        self._f = open(self.fp, "a", encoding="utf-8", buffering=1)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._f is not None:
            self._f.flush()
            self._f.close()
            self._f = None

    def write(self, obj: Dict[str, Any]) -> None:
        assert self._f is not None, "JsonlWriter must be used as a context manager"
        self._f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._n += 1
        if self.flush_every > 0 and (self._n % self.flush_every) == 0:
            self._f.flush()

    def close(self) -> None:
        self._f.close()
    