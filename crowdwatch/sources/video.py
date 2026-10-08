"""Video from a file, a webcam or a network camera."""

from __future__ import annotations

import threading
import time
from typing import Optional, Union

from .base import Frame

_STREAM_PREFIXES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://", "udp://", "tcp://")


def is_live(uri: Union[str, int]) -> bool:
    if isinstance(uri, int):
        return True
    text = str(uri)
    return text.isdigit() or text.lower().startswith(_STREAM_PREFIXES)


class VideoSource:
    """Reads frames at roughly ``fps`` frames per second of video time.

    Recordings are read in order (skipping frames to hit the target rate) and
    carry the recording's own clock, so results are repeatable. Live cameras
    are read on a background thread that always keeps only the newest frame;
    if analysis falls behind, stale frames are dropped rather than queued.
    """

    def __init__(self, uri: Union[str, int], fps: float = 10.0, loop: bool = True):
        import cv2

        self._cv2 = cv2
        self.uri = int(uri) if isinstance(uri, str) and uri.isdigit() else uri
        self.live = is_live(uri)
        self.loop = loop and not self.live
        self.target_fps = fps
        self.speed = 1.0
        self.rewound = False            # set when a recording starts over
        self.finished = False
        self._offset = 0.0
        self._index = 0
        self._cap = self._open()
        self.size = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        src_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.source_fps = src_fps if src_fps and 1 <= src_fps <= 240 else 25.0
        self.stride = 1 if self.live else max(1, int(round(self.source_fps / fps)))

        self._latest = None
        self._latest_id = 0
        self._served_id = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._t0 = time.monotonic()
        if self.live:
            ok, first = self._cap.read()
            if ok:
                self.size = (first.shape[1], first.shape[0])
                self._latest, self._latest_id = first, 1
            threading.Thread(target=self._pump, daemon=True, name="camera-reader").start()

    def _open(self):
        cap = self._cv2.VideoCapture(self.uri)
        if not cap.isOpened():
            raise RuntimeError(
                f"could not open video source {self.uri!r}. Check the path or camera address."
            )
        return cap

    # ---------------------------------------------------------------- live

    def _pump(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            ok, image = self._cap.read()
            if not ok:
                # camera dropped: wait, then reconnect
                time.sleep(backoff)
                backoff = min(backoff * 2, 15.0)
                try:
                    self._cap.release()
                    self._cap = self._open()
                except RuntimeError:
                    pass
                continue
            backoff = 1.0
            with self._lock:
                self._latest, self._latest_id = image, self._latest_id + 1

    def _read_live(self) -> Optional[Frame]:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not self._stop.is_set():
            with self._lock:
                if self._latest_id != self._served_id:
                    self._served_id = self._latest_id
                    return Frame(t=time.monotonic() - self._t0, image=self._latest)
            time.sleep(0.005)
        return Frame(t=time.monotonic() - self._t0, image=None)   # no picture, keep the clock going

    # ---------------------------------------------------------------- file

    def _read_file(self) -> Optional[Frame]:
        for _ in range(self.stride - 1):
            if not self._cap.grab():
                break
            self._index += 1
        ok, image = self._cap.read()
        if not ok:
            if not self.loop or self._index == 0:
                self.finished = True
                return None
            self._offset += self._index / self.source_fps
            self._index = 0
            self._cap.release()
            self._cap = self._open()
            self.rewound = True
            ok, image = self._cap.read()
            if not ok:
                self.finished = True
                return None
        self._index += 1
        return Frame(t=self._offset + self._index / self.source_fps, image=image)

    def read(self) -> Optional[Frame]:
        return self._read_live() if self.live else self._read_file()

    def close(self) -> None:
        self._stop.set()
        try:
            self._cap.release()
        except Exception:
            pass
