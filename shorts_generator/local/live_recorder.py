"""Record a live stream to local part files while it is on air.

ViralClip's edge over a plain clipper is that it cuts clips *during* the
stream, not only after it ends — and that is the point here too. A live is
recorded with ffmpeg's segment muxer, so the recording lands as one finished
file per few minutes (parte-001.mp4, parte-002.mp4, ...) instead of a single
file that only exists once the stream is over. Each finished part is handed
straight to the normal clip pipeline, so clips appear while the live is
still going.

The stream URL comes from yt-dlp (same cookie/access handling as the
downloader, see yt_access), which is what makes YouTube, TikTok, Twitch,
Kick and Facebook lives all work through one path. ffmpeg does the actual
recording with `-c copy`: nothing is re-encoded, so the parts are exact
copies of what aired and the recording costs almost no CPU.
"""
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.parse import urlparse

from . import yt_access


def platform_of(url: str) -> str:
    """Which service a live link is on, from its host."""
    host = (urlparse(url or "").netloc or "").lower()
    host = host[4:] if host.startswith("www.") else host
    for name, frag in (
        ("youtube", ("youtube.com", "youtu.be")),
        ("tiktok", ("tiktok.com",)),
        ("twitch", ("twitch.tv",)),
        ("kick", ("kick.com",)),
        ("facebook", ("facebook.com", "fb.watch", "fb.com")),
    ):
        if any(host == f or host.endswith("." + f) for f in frag):
            return name
    return "other"


def _import_ytdlp():
    try:
        import yt_dlp  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "yt-dlp is required to record a live. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e
    return yt_dlp


def _best_stream_urls(info: Dict) -> Dict[str, Optional[str]]:
    """Direct media URLs for a live: a combined stream, or video + audio.

    yt-dlp's `url` is usually the manifest (HLS/DASH), which ffmpeg reads
    directly, so no download happens here — this only picks which URL(s) to
    hand to ffmpeg.
    """
    formats = [f for f in (info.get("formats") or []) if f.get("url")]
    if not formats:
        direct = info.get("url")
        return {"video_url": direct, "audio_url": None} if direct else {}

    def is_video(f):
        return f.get("vcodec") not in (None, "none")

    def is_audio(f):
        return f.get("acodec") not in (None, "none") and not is_video(f)

    combined = [f for f in formats if is_video(f) and is_audio(f)]
    if combined:
        # Best picture up to 1080p: above that the parts are huge and the
        # clips are vertical anyway.
        combined.sort(key=lambda f: (f.get("height") or 0))
        pick = next((f for f in combined if (f.get("height") or 0) <= 1080),
                    combined[-1])
        return {"video_url": pick["url"], "audio_url": None}

    video = [f for f in formats if is_video(f)]
    audio = [f for f in formats if is_audio(f)]
    if not video:
        return {}
    video.sort(key=lambda f: (f.get("height") or 0))
    v = next((f for f in video if (f.get("height") or 0) <= 1080), video[-1])
    a = None
    if audio:
        audio.sort(key=lambda f: (f.get("abr") or 0))
        a = audio[-1]["url"]
    return {"video_url": v["url"], "audio_url": a}


def resolve_live(url: str) -> Dict:
    """What a live link is, and the direct stream URL(s) to record.

    Raises ValueError with a plain-language reason when the link is not a
    live that can be recorded (not live, not found, needs a login).
    """
    yt_dlp = _import_ytdlp()
    url = (url or "").strip()
    if not url:
        raise ValueError("Pega el enlace del live primero.")
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True}

    def once(ydl):
        return ydl.extract_info(url, download=False)

    try:
        info = yt_access.run(yt_dlp, opts, once)
    except Exception as e:
        msg = str(e)
        if yt_access.needs_an_account(e):
            raise ValueError(
                "Ese live pide iniciar sesión para verlo. Abre el live en tu "
                "navegador con tu cuenta y vuelve a intentarlo — la app usa "
                "las cookies de ese navegador.") from e
        if "not live" in msg.lower() or "is not live" in msg.lower():
            raise ValueError("Ese enlace no está en vivo ahora mismo.") from e
        raise ValueError(f"No se pudo abrir el live: {msg.splitlines()[0] if msg else 'error desconocido'}") from e

    if not info:
        raise ValueError("No se pudo leer ese enlace.")
    if info.get("is_live") is False:
        raise ValueError(
            "Ese video no es un live en curso — es un video ya publicado. "
            "Para sacar clips de un video usa «Sacar clips de YouTube».")
    urls = _best_stream_urls(info)
    if not urls.get("video_url"):
        raise ValueError("No se encontró el flujo de video de ese live.")
    title = info.get("title") or info.get("uploader") or url
    return {
        "title": str(title),
        "platform": platform_of(url),
        "is_live": bool(info.get("is_live")),
        "video_url": urls["video_url"],
        "audio_url": urls.get("audio_url"),
    }


class LiveRecorder:
    """ffmpeg, recording a live into one finished file per few minutes.

    `on_part` is called with each part file as soon as it is closed, so the
    caller can cut clips from it while the stream is still going. The part
    being written is not ready until the next one starts (or the recording
    stops), which the watcher works out from the file numbering.
    """

    def __init__(self, parts_dir: str, segment_seconds: int,
                 on_part: Callable[[str, int], None],
                 on_ended: Optional[Callable[[Optional[str]], None]] = None,
                 start_number: int = 1):
        self.parts_dir = Path(parts_dir)
        self.segment_seconds = max(2, int(segment_seconds))
        self.on_part = on_part
        self.on_ended = on_ended
        self.start_number = max(1, int(start_number))
        self.proc: Optional[subprocess.Popen] = None
        self._stop = threading.Event()
        self._watcher: Optional[threading.Thread] = None
        self._reported: set = set()
        self.error: Optional[str] = None

    def _part_name(self, n: int) -> str:
        return f"parte-{n:03d}.mp4"

    def start(self, video_url: str, audio_url: Optional[str] = None) -> None:
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg", "-v", "error",
               "-reconnect", "1", "-reconnect_streamed", "1",
               "-reconnect_delay_max", "5",
               "-i", video_url]
        if audio_url:
            cmd += ["-reconnect", "1", "-reconnect_streamed", "1",
                    "-reconnect_delay_max", "5", "-i", audio_url]
            maps = ["-map", "0:v", "-map", "1:a"]
        else:
            maps = ["-map", "0"]
        cmd += maps + [
            "-c", "copy",
            "-f", "segment",
            "-segment_time", str(self.segment_seconds),
            "-segment_format", "mp4",
            "-reset_timestamps", "1",
            "-segment_start_number", str(self.start_number),
            str(self.parts_dir / "parte-%03d.mp4"),
        ]
        print(f"[live] recording into {self.parts_dir} "
              f"(1 part every {self.segment_seconds // 60} min)", flush=True)
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self._watcher = threading.Thread(target=self._watch, daemon=True,
                                         name="live-watch")
        self._watcher.start()

    def _ready_parts(self) -> List[int]:
        """Part numbers whose files are closed and ready to cut from."""
        nums = []
        for p in self.parts_dir.glob("parte-*.mp4"):
            m = re.match(r"parte-(\d+)\.mp4$", p.name)
            if m:
                nums.append(int(m.group(1)))
        if not nums:
            return []
        nums.sort()
        # A part is closed when a later one exists, or when ffmpeg has exited.
        done = self.proc is not None and self.proc.poll() is not None
        return [n for n in nums if done or n < nums[-1]]

    def _watch(self) -> None:
        while not self._stop.is_set():
            try:
                for n in self._ready_parts():
                    if n not in self._reported:
                        self._reported.add(n)
                        self.on_part(str(self.parts_dir / self._part_name(n)), n)
            except Exception as e:  # noqa: BLE001 - the watcher must not die
                print(f"[live] watcher: {e}", flush=True)
            if self.proc is not None and self.proc.poll() is not None:
                break
            self._stop.wait(2.0)
        # One last sweep: the final part closes when ffmpeg exits.
        try:
            for n in self._ready_parts():
                if n not in self._reported:
                    self._reported.add(n)
                    self.on_part(str(self.parts_dir / self._part_name(n)), n)
        except Exception as e:  # noqa: BLE001
            print(f"[live] watcher: {e}", flush=True)
        err = None
        if self.proc is not None:
            rc = self.proc.poll()
            if rc not in (0, None):
                try:
                    err = (self.proc.stderr.read() or b"").decode("utf-8", "replace")
                    err = (err.strip().splitlines() or ["ffmpeg exited"])[
                        -1][:300]
                except Exception:
                    err = "ffmpeg exited"
        self.error = err
        if self.on_ended:
            self.on_ended(err)

    def stop(self, timeout: float = 20.0) -> None:
        """Stop the recording and finalize the part being written.

        SIGINT lets ffmpeg close the file cleanly, so the last part is a
        valid mp4 — killing it would leave a file with no index that nothing
        can read.
        """
        self._stop.set()
        proc = self.proc
        if proc is not None and proc.poll() is None:
            try:
                proc.send_signal(signal.SIGINT)
            except Exception:
                pass
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)
        if self._watcher is not None:
            self._watcher.join(timeout=10.0)

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None
