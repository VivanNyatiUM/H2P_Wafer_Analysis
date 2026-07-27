"""Small dependency-free console progress bars used by H2P batch stages."""

from __future__ import annotations

import math
import sys
import time
from typing import Optional


class ProgressBar:
    """A throttled, single-line progress bar with elapsed time and ETA."""

    def __init__(
        self,
        label: str,
        total: int,
        *,
        width: int = 28,
        min_interval: float = 0.20,
    ) -> None:
        self.label = str(label)
        self.total = max(0, int(total))
        self.width = max(10, int(width))
        self.min_interval = max(0.0, float(min_interval))
        self.started = time.perf_counter()
        self.last_rendered = 0.0
        self.last_current = -1
        self.last_line_length = 0
        self.current = 0
        self.extra = ""
        self._tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self.update(0, force=True)

    @staticmethod
    def _format_minutes(seconds: Optional[float]) -> str:
        if seconds is None or not math.isfinite(seconds) or seconds < 0:
            return "  --"
        return f"{seconds / 60.0:4.1f}m"

    def _line(self, current: int, extra: str) -> str:
        elapsed = max(0.0, time.perf_counter() - self.started)
        if self.total > 0:
            fraction = min(1.0, max(0.0, current / self.total))
            filled = int(round(self.width * fraction))
            eta = elapsed * (self.total - current) / current if current > 0 else None
            count = f"{current}/{self.total}"
            percent = f"{100.0 * fraction:5.1f}%"
        else:
            fraction = 1.0
            filled = self.width
            eta = 0.0
            count = "0/0"
            percent = "100.0%"
        bar = "#" * filled + "-" * (self.width - filled)
        line = (
            f"[{self.label}] |{bar}| {count:>9} {percent} "
            f"elapsed {self._format_minutes(elapsed)} ETA {self._format_minutes(eta)}"
        )
        if extra:
            line += f" | {extra}"
        return line

    def update(self, current: int, *, extra: str = "", force: bool = False) -> None:
        now = time.perf_counter()
        current = max(0, min(int(current), self.total if self.total > 0 else 0))
        self.current = current
        self.extra = str(extra or "")
        if not force and current < self.total:
            if now - self.last_rendered < self.min_interval:
                return
            if not self._tty and self.total > 0:
                # Keep redirected logs readable instead of emitting thousands of lines.
                step = max(1, self.total // 20)
                if current != self.total and current % step != 0:
                    return
        line = self._line(current, self.extra)
        if self._tty:
            padding = " " * max(0, self.last_line_length - len(line))
            print("\r" + line + padding, end="", flush=True)
            self.last_line_length = len(line)
        else:
            if current != self.last_current or force:
                print(line, flush=True)
        self.last_current = current
        self.last_rendered = now

    def status(self, extra: str) -> None:
        self.update(self.current, extra=extra, force=True)

    def done(self, *, extra: str = "") -> None:
        self.update(self.total, extra=extra, force=True)
        if self._tty:
            print(flush=True)
