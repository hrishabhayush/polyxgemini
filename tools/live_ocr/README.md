# Live NCAA Subtitle OCR

Low-latency subtitle OCR pipeline for livestream windows on macOS.

## What It Does

- Captures a target screen region continuously with FFmpeg.
- Crops to subtitle area (bottom band by default).
- Runs PaddleOCR on preprocessed subtitle frames.
- Emits deduplicated live transcript events as JSON lines.
- Optionally writes `.srt` entries from the same event stream.

## Prerequisites

- macOS with Screen Recording permission enabled for your terminal/python host.
- `ffmpeg` installed and available on `PATH`.
- Python 3.10 or 3.11 recommended (Paddle wheels are not reliably available on 3.12).

Install Python deps:

```bash
pip install -r tools/live_ocr/requirements.txt
```

## Run

```bash
python tools/live_ocr/runner.py --config tools/live_ocr/config.yaml --window-title "NCAA"
```

If the window title lookup fails, the configured `capture_rect` is used.

### First-Time ROI Calibration

```bash
python tools/live_ocr/runner.py --config tools/live_ocr/config.yaml --calibrate
```

This opens a selection UI and persists the selected ROI into `roi.locked_rect` in config.
Subsequent normal runs use the saved ROI automatically.

To clear a previously saved ROI:

```bash
python tools/live_ocr/runner.py --config tools/live_ocr/config.yaml --clear-saved-roi
```

## Output Format

By default, stdout emits newline-delimited JSON:

```json
{"timestamp": 1711234567.12, "frame_id": 42, "text": "Final score update", "confidence": 0.91}
```

Your ML consumer can subscribe to stdout (or a redirected file/pipe) and recompute probabilities on each event.

## Notes

- No manual start/stop recording is needed per chunk. The process continuously samples frames in-memory.
- For lower CPU, reduce `capture.fps` and/or `capture.resize_width` in config.
- If OCR misses lines, tune `roi.bottom_fraction`, `postprocess.min_confidence`, and OCR thresholds.

## Troubleshooting

- If startup fails with `No module named 'langchain.docstore'`, reinstall pinned deps:

```bash
pip install -r tools/live_ocr/requirements.txt --upgrade
```

- If you still see Paddle source host checks during startup, this can be skipped:

```bash
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
```

- If startup fails with `No module named 'paddle'`, your interpreter is missing `paddlepaddle`
  (commonly with Python 3.12). Create a Python 3.11 virtualenv and reinstall:

```bash
python3.11 -m venv .venv-ocr
source .venv-ocr/bin/activate
pip install --upgrade pip
pip install -r tools/live_ocr/requirements.txt
pip install paddlepaddle
```

