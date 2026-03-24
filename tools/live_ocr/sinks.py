from __future__ import annotations

import json
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Optional

from postprocess import TranscriptEvent


class LiveJsonSink:
    def emit(self, event: TranscriptEvent) -> None:
        print(json.dumps(asdict(event), ensure_ascii=True), flush=True)


class SrtSink:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._idx = 0
        self._last_event: Optional[TranscriptEvent] = None

    @staticmethod
    def _format_ts(seconds: float) -> str:
        td = timedelta(seconds=max(0.0, seconds))
        total = int(td.total_seconds())
        ms = int((td.total_seconds() - total) * 1000)
        hours = total // 3600
        minutes = (total % 3600) // 60
        secs = total % 60
        return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"

    def emit(self, event: TranscriptEvent) -> None:
        self._idx += 1
        start = event.timestamp
        end = event.timestamp + 1.5
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"{self._idx}\n")
            handle.write(f"{self._format_ts(start)} --> {self._format_ts(end)}\n")
            handle.write(f"{event.text}\n\n")
        self._last_event = event

