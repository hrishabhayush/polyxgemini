from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from typing import Generator, Optional

import numpy as np


@dataclass
class CaptureRect:
    x: int
    y: int
    width: int
    height: int


class CaptureError(RuntimeError):
    pass


def resolve_window_rect(window_title: str) -> Optional[CaptureRect]:
    """
    Resolve a frontmost macOS window by partial title match using AppleScript.
    Returns None when not found.
    """
    if not window_title.strip():
        return None

    applescript = f"""
set targetTitle to "{window_title}"
tell application "System Events"
    set procList to application processes whose background only is false
    repeat with p in procList
        repeat with w in windows of p
            try
                set winTitle to name of w
                if winTitle contains targetTitle then
                    set winPos to position of w
                    set winSize to size of w
                    return "{{\\"x\\":" & item 1 of winPos & ",\\"y\\":" & item 2 of winPos & ",\\"width\\":" & item 1 of winSize & ",\\"height\\":" & item 2 of winSize & "}}"
                end if
            end try
        end repeat
    end repeat
end tell
return ""
""".strip()

    result = subprocess.run(
        ["osascript", "-e", applescript],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None

    output = result.stdout.strip()
    if not output:
        return None
    try:
        payload = json.loads(output)
        return CaptureRect(
            x=int(payload["x"]),
            y=int(payload["y"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
        )
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


class FFmpegFrameCapture:
    def __init__(
        self,
        ffmpeg_bin: str,
        fps: float,
        screen_index: str,
        pixel_format: str,
        capture_rect: CaptureRect,
        resize_width: int = 0,
    ) -> None:
        self.ffmpeg_bin = ffmpeg_bin
        self.fps = fps
        self.screen_index = screen_index
        self.pixel_format = pixel_format
        self.capture_rect = capture_rect
        self.resize_width = resize_width
        self._process: Optional[subprocess.Popen] = None

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        width = self.capture_rect.width
        height = self.capture_rect.height
        if self.resize_width and self.resize_width > 0:
            scale = self.resize_width / float(width)
            width = int(self.resize_width)
            height = int(height * scale)
        return (height, width, 3)

    def _build_command(self) -> list[str]:
        vf_chain = [
            (
                f"crop={self.capture_rect.width}:{self.capture_rect.height}:"
                f"{self.capture_rect.x}:{self.capture_rect.y}"
            )
        ]
        if self.resize_width and self.resize_width > 0:
            vf_chain.append(f"scale={self.resize_width}:-2")

        return [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "avfoundation",
            "-framerate",
            str(self.fps),
            "-i",
            f"{self.screen_index}:none",
            "-vf",
            ",".join(vf_chain),
            "-pix_fmt",
            self.pixel_format,
            "-f",
            "rawvideo",
            "pipe:1",
        ]

    def start(self) -> None:
        if self._process is not None:
            return
        self._validate_ffmpeg()
        cmd = self._build_command()
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _validate_ffmpeg(self) -> None:
        check = subprocess.run(
            [self.ffmpeg_bin, "-version"],
            capture_output=True,
            text=True,
            check=False,
        )
        if check.returncode != 0:
            raise CaptureError(
                f"ffmpeg binary '{self.ffmpeg_bin}' is not available. Install ffmpeg first."
            )

    def close(self) -> None:
        if self._process is None:
            return
        self._process.terminate()
        self._process.wait(timeout=2)
        self._process = None

    def frames(self) -> Generator[np.ndarray, None, None]:
        if self._process is None:
            self.start()
        assert self._process is not None
        assert self._process.stdout is not None

        h, w, c = self.frame_shape
        frame_bytes = h * w * c
        while True:
            data = self._process.stdout.read(frame_bytes)
            if not data or len(data) < frame_bytes:
                stderr_output = ""
                if self._process.stderr is not None:
                    stderr_output = self._process.stderr.read().decode("utf-8", "ignore")
                raise CaptureError(
                    "FFmpeg capture stopped. Check macOS Screen Recording "
                    f"permissions and ffmpeg input index. stderr={shlex.quote(stderr_output)}"
                )
            frame = np.frombuffer(data, dtype=np.uint8).reshape((h, w, c))
            # Normalize to BGR when ffmpeg output changes by platform.
            if self.pixel_format.lower() == "rgb24":
                frame = frame[:, :, ::-1]
            yield frame

