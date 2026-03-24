from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import yaml

LOGGER = logging.getLogger("live_ocr")


@dataclass
class PipelineConfig:
    raw: dict

    @classmethod
    def load(cls, path: str) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return cls(raw=payload)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.raw, handle, sort_keys=False)


class LiveOcrRunner:
    def __init__(
        self,
        cfg: PipelineConfig,
        cfg_path: str,
        override_window_title: str = "",
        clear_saved_roi: bool = False,
    ) -> None:
        from ocr_engine import PaddleSubtitleOCR
        from postprocess import SubtitlePostProcessor
        from sinks import LiveJsonSink, SrtSink

        self.pipeline_config = cfg
        self.cfg = cfg.raw
        self.cfg_path = cfg_path
        self.stop = False
        self.override_window_title = override_window_title
        self.clear_saved_roi = clear_saved_roi

        if self.clear_saved_roi:
            self._clear_persisted_roi()

        self.live_sink = LiveJsonSink() if self.cfg["output"]["emit_stdout_json"] else None
        self.srt_sink = SrtSink(self.cfg["output"]["srt_path"]) if self.cfg["output"]["save_srt"] else None

        ocr_cfg = self.cfg["ocr"]
        self.ocr = PaddleSubtitleOCR(
            language=ocr_cfg["language"],
            use_angle_cls=ocr_cfg["use_angle_cls"],
            det_db_box_thresh=float(ocr_cfg["det_db_box_thresh"]),
            det_db_thresh=float(ocr_cfg["det_db_thresh"]),
            rec_score_thresh=float(ocr_cfg["rec_score_thresh"]),
        )

        pp_cfg = self.cfg["postprocess"]
        self.postprocess = SubtitlePostProcessor(
            min_confidence=float(pp_cfg["min_confidence"]),
            min_characters=int(pp_cfg["min_characters"]),
            stable_hits_required=int(pp_cfg["stable_hits_required"]),
            duplicate_silence_seconds=float(pp_cfg["duplicate_silence_seconds"]),
        )

    def _clear_persisted_roi(self) -> None:
        roi_cfg = self.cfg.setdefault("roi", {})
        if roi_cfg.get("locked_rect"):
            roi_cfg["locked_rect"] = None
            self.pipeline_config.save(self.cfg_path)
            LOGGER.info("Cleared persisted ROI from config file")

    def _persist_roi(self, roi_spec: Any) -> None:
        roi_cfg = self.cfg.setdefault("roi", {})
        roi_cfg["locked_rect"] = {
            "x": int(roi_spec.x),
            "y": int(roi_spec.y),
            "width": int(roi_spec.width),
            "height": int(roi_spec.height),
        }
        roi_cfg["locked_rect_frame"] = {
            "width": int(getattr(roi_spec, "frame_width", 0)),
            "height": int(getattr(roi_spec, "frame_height", 0)),
        }
        self.pipeline_config.save(self.cfg_path)
        LOGGER.info(
            "Persisted ROI to config at %s: x=%d y=%d w=%d h=%d",
            self.cfg_path,
            roi_spec.x,
            roi_spec.y,
            roi_spec.width,
            roi_spec.height,
        )

    def _clamp_roi_to_frame(self, roi_spec: Any, frame: Any) -> Optional[Any]:
        from roi import RoiSpec

        frame_h, frame_w = frame.shape[:2]
        x = max(0, int(roi_spec.x))
        y = max(0, int(roi_spec.y))
        w = int(roi_spec.width)
        h = int(roi_spec.height)
        if x >= frame_w or y >= frame_h or w <= 0 or h <= 0:
            return None
        w = min(w, frame_w - x)
        h = min(h, frame_h - y)
        if w <= 0 or h <= 0:
            return None
        return RoiSpec(x=x, y=y, width=w, height=h)

    def _resolve_startup_roi(self, frame: Any, calibrate: bool) -> Any:
        from roi import RoiSpec, calibrate_band, subtitle_band_roi

        roi_cfg = self.cfg["roi"]
        if calibrate:
            roi_spec = calibrate_band(frame)
            # Store capture frame dimensions for future scaling checks.
            roi_spec.frame_width = frame.shape[1]
            roi_spec.frame_height = frame.shape[0]
            self._persist_roi(roi_spec)
            return roi_spec

        locked = roi_cfg.get("locked_rect")
        if locked:
            locked_roi = RoiSpec(
                x=int(locked["x"]),
                y=int(locked["y"]),
                width=int(locked["width"]),
                height=int(locked["height"]),
            )
            clamped = self._clamp_roi_to_frame(locked_roi, frame)
            if clamped is not None:
                return clamped
            LOGGER.warning("Saved locked ROI is out of current frame bounds; clearing it")
            roi_cfg["locked_rect"] = None
            self.pipeline_config.save(self.cfg_path)

        return subtitle_band_roi(
            frame,
            bottom_fraction=float(roi_cfg["bottom_fraction"]),
            horizontal_margin_fraction=float(roi_cfg["horizontal_margin_fraction"]),
        )

    def _resolve_capture_rect(self) -> Any:
        from capture import CaptureRect, resolve_window_rect

        capture_cfg = self.cfg["capture"]
        title = self.override_window_title or capture_cfg.get("window_title", "")
        if title:
            resolved = resolve_window_rect(title)
            if resolved:
                LOGGER.info("Resolved window '%s' to rect=%s", title, resolved)
                return resolved
            LOGGER.warning("Could not resolve window title '%s'; using config capture_rect", title)

        rect_cfg = capture_cfg["capture_rect"]
        return CaptureRect(
            x=int(rect_cfg["x"]),
            y=int(rect_cfg["y"]),
            width=int(rect_cfg["width"]),
            height=int(rect_cfg["height"]),
        )

    def run(self, calibrate: bool = False) -> None:
        from capture import FFmpegFrameCapture
        from roi import RoiSpec, crop, preprocess_for_subtitles

        capture_cfg = self.cfg["capture"]
        cap = FFmpegFrameCapture(
            ffmpeg_bin=capture_cfg["ffmpeg_bin"],
            fps=float(capture_cfg["fps"]),
            screen_index=str(capture_cfg["screen_index"]),
            pixel_format=str(capture_cfg["pixel_format"]),
            capture_rect=self._resolve_capture_rect(),
            resize_width=int(capture_cfg["resize_width"]),
        )

        signal.signal(signal.SIGINT, self._handle_stop)
        signal.signal(signal.SIGTERM, self._handle_stop)

        frame_id = 0
        roi_spec: Optional[RoiSpec] = None
        start = time.time()
        ocr_time_total = 0.0
        ocr_calls = 0
        last_report_frame = 0
        prev_small_gray = None
        min_change_ratio = 0.006
        try:
            for frame in cap.frames():
                frame_id += 1
                if roi_spec is None:
                    roi_spec = self._resolve_startup_roi(frame=frame, calibrate=calibrate)
                    LOGGER.info("Using ROI x=%d y=%d w=%d h=%d", roi_spec.x, roi_spec.y, roi_spec.width, roi_spec.height)

                roi_frame = crop(frame, roi_spec)
                small_gray = cv2.cvtColor(cv2.resize(roi_frame, (160, 40)), cv2.COLOR_BGR2GRAY)
                if prev_small_gray is not None:
                    diff = cv2.absdiff(small_gray, prev_small_gray)
                    change_ratio = float((diff > 12).mean())
                    if change_ratio < min_change_ratio:
                        if self.stop:
                            break
                        continue
                prev_small_gray = small_gray

                preprocessed = preprocess_for_subtitles(roi_frame)
                t0 = time.perf_counter()
                ocr_lines = self.ocr.run(preprocessed)
                ocr_time_total += time.perf_counter() - t0
                ocr_calls += 1
                event = self.postprocess.process(frame_id=frame_id, ocr_lines=ocr_lines)
                if event:
                    if self.live_sink:
                        self.live_sink.emit(event)
                    if self.srt_sink:
                        self.srt_sink.emit(event)

                if frame_id - last_report_frame >= 30:
                    elapsed = time.time() - start
                    run_fps = frame_id / elapsed if elapsed else 0.0
                    avg_ocr_ms = (ocr_time_total / max(1, ocr_calls)) * 1000.0
                    LOGGER.info(
                        "progress frames=%d fps=%.2f ocr_ms=%.1f emitted=%d drop_low=%d drop_unstable=%d drop_dup=%d",
                        frame_id,
                        run_fps,
                        avg_ocr_ms,
                        self.postprocess.emitted,
                        self.postprocess.filtered_low_quality,
                        self.postprocess.filtered_unstable,
                        self.postprocess.filtered_duplicate,
                    )
                    last_report_frame = frame_id

                if self.stop:
                    break
        finally:
            cap.close()
            elapsed = time.time() - start
            fps = frame_id / elapsed if elapsed else 0.0
            LOGGER.info("Stopped. frames=%d elapsed=%.2fs avg_fps=%.2f", frame_id, elapsed, fps)

    def _handle_stop(self, *_args) -> None:
        self.stop = True


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live OCR subtitle pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("config.yaml")),
        help="Path to pipeline YAML config",
    )
    parser.add_argument(
        "--window-title",
        type=str,
        default="",
        help="Optional macOS window title contains-match for auto capture bounds",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Open manual ROI picker before streaming OCR",
    )
    parser.add_argument(
        "--clear-saved-roi",
        action="store_true",
        help="Remove persisted ROI from config and fall back to dynamic ROI",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args(argv)
    cfg = PipelineConfig.load(args.config)
    runner = LiveOcrRunner(
        cfg=cfg,
        cfg_path=args.config,
        override_window_title=args.window_title,
        clear_saved_roi=args.clear_saved_roi,
    )
    runner.run(calibrate=args.calibrate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

