from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional

from ocr_engine import OcrResult


@dataclass
class TranscriptEvent:
    timestamp: float
    frame_id: int
    text: str
    confidence: float


class SubtitlePostProcessor:
    def __init__(
        self,
        min_confidence: float,
        min_characters: int,
        stable_hits_required: int,
        duplicate_silence_seconds: float,
    ) -> None:
        self.min_confidence = min_confidence
        self.min_characters = min_characters
        self.stable_hits_required = stable_hits_required
        self.duplicate_silence_seconds = duplicate_silence_seconds

        self._candidate_text: str = ""
        self._candidate_hits = 0
        self._last_emitted_text: str = ""
        self._last_emitted_ts = 0.0
        self.filtered_low_quality = 0
        self.filtered_unstable = 0
        self.filtered_duplicate = 0
        self.emitted = 0

    @staticmethod
    def normalize(text: str) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        return normalized

    def _compose_line(self, ocr_lines: list[OcrResult]) -> tuple[str, float]:
        filtered = [x for x in ocr_lines if x.confidence >= self.min_confidence and x.text]
        if not filtered:
            return "", 0.0
        text = " ".join([self.normalize(x.text) for x in filtered]).strip()
        confidence = sum([x.confidence for x in filtered]) / len(filtered)
        return text, confidence

    def process(self, frame_id: int, ocr_lines: list[OcrResult]) -> Optional[TranscriptEvent]:
        now = time.time()
        text, confidence = self._compose_line(ocr_lines)
        if len(text) < self.min_characters:
            self.filtered_low_quality += 1
            return None

        if text == self._candidate_text:
            self._candidate_hits += 1
        else:
            self._candidate_text = text
            self._candidate_hits = 1

        if self._candidate_hits < self.stable_hits_required:
            self.filtered_unstable += 1
            return None
        if (
            text == self._last_emitted_text
            and now - self._last_emitted_ts < self.duplicate_silence_seconds
        ):
            self.filtered_duplicate += 1
            return None

        self._last_emitted_text = text
        self._last_emitted_ts = now
        self.emitted += 1
        return TranscriptEvent(
            timestamp=now,
            frame_id=frame_id,
            text=text,
            confidence=confidence,
        )

