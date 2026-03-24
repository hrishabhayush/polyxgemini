from __future__ import annotations

from dataclasses import dataclass
from typing import List

try:
    from paddleocr import PaddleOCR
except ModuleNotFoundError as exc:
    if exc.name == "paddle":
        raise ModuleNotFoundError(
            "PaddleOCR requires paddlepaddle, which is not installed for this interpreter.\n"
            "Use Python 3.10 or 3.11 in a virtualenv, then run:\n"
            "  pip install -r tools/live_ocr/requirements.txt\n"
            "  pip install paddlepaddle"
        ) from exc
    raise ModuleNotFoundError(
        "PaddleOCR import failed. Reinstall dependencies with:\n"
        "  pip install -r tools/live_ocr/requirements.txt\n"
        "If you still see 'langchain.docstore' errors, run:\n"
        "  pip install \"langchain<0.2\" \"paddleocr>=2.7,<3.0\""
    ) from exc


@dataclass
class OcrResult:
    text: str
    confidence: float


class PaddleSubtitleOCR:
    def __init__(
        self,
        language: str,
        use_angle_cls: bool,
        det_db_box_thresh: float,
        det_db_thresh: float,
        rec_score_thresh: float,
    ) -> None:
        self.engine = PaddleOCR(
            lang=language,
            use_angle_cls=use_angle_cls,
            det_db_box_thresh=det_db_box_thresh,
            det_db_thresh=det_db_thresh,
            rec_score_thresh=rec_score_thresh,
            show_log=False,
        )

    def run(self, image) -> List[OcrResult]:
        results = self.engine.ocr(image, cls=False)
        if not results or not results[0]:
            return []

        out: List[OcrResult] = []
        for line in results[0]:
            text, confidence = line[1]
            if text is None:
                continue
            out.append(OcrResult(text=text.strip(), confidence=float(confidence)))
        return out

