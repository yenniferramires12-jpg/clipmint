"""Live recordings: one live at a time, cut into parts, clips while it airs.

A live is recorded in parts (see shorts_generator/local/live_recorder.py)
and every finished part goes straight into the normal clip pipeline as its
own job, so clips appear while the stream is still going — that is the whole
difference from cutting a finished VOD afterwards. One live records at a
time: two ffmpeg captures would fight over the machine a render already
wants.

Records live in a small ledger (lives.json in the config dir) so "Tus Lives"
survives a restart. A live the app did not get to stop cleanly comes back
as "interrupted", with the parts it did record still on disk and a way to
cut clips from the parts that never got a job.
"""
import copy
import json
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from shorts_generator import user_config
from shorts_generator.layout_spec import LayoutSpec
from shorts_generator.local import live_recorder

from .jobs import STORE

LEDGER_NAME = "lives.json"


@dataclass
class LiveRecord:
    id: str
    url: str
    platform: str = "other"
    title: str = ""
    status: str = "starting"  # starting | recording | ended | stopped | error | interrupted
    started_at: float = 0.0
    ended_at: Optional[float] = None
    parts_dir: str = ""
    clips_per_part: int = 2
    part_minutes: int = 5
    part_jobs: List[Dict] = field(default_factory=list)  # {part, job_id}
    clips_count: int = 0
    error: Optional[str] = None
    # The LayoutSpec the live's parts render with, as a dict — persisted so
    # a live interrupted by closing the app can still cut its remaining parts
    # afterwards, when the in-memory spec is long gone.
    spec: Optional[Dict] = None

    def snapshot(self) -> Dict:
        return {
            "id": self.id,
            "url": self.url,
            "platform": self.platform,
            "title": self.title,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "elapsed": round((self.ended_at or time.time()) - self.started_at, 1)
                       if self.started_at else 0,
            "parts_dir": self.parts_dir,
            "clips_per_part": self.clips_per_part,
            "part_minutes": self.part_minutes,
            "part_jobs": list(self.part_jobs),
            "clips_count": self.clips_count,
            "error": self.error,
            "spec": self.spec,
        }

    def to_ledger(self) -> Dict:
        d = self.snapshot()
        d.pop("elapsed", None)
        return d

    @classmethod
    def from_ledger(cls, d: Dict) -> "LiveRecord":
        return cls(
            id=str(d.get("id") or ""),
            url=str(d.get("url") or ""),
            platform=str(d.get("platform") or "other"),
            title=str(d.get("title") or ""),
            status=str(d.get("status") or "ended"),
            started_at=float(d.get("started_at") or 0),
            ended_at=d.get("ended_at"),
            parts_dir=str(d.get("parts_dir") or ""),
            clips_per_part=int(d.get("clips_per_part") or 2),
            part_minutes=int(d.get("part_minutes") or 5),
            part_jobs=list(d.get("part_jobs") or []),
            clips_count=int(d.get("clips_count") or 0),
            error=d.get("error"),
            spec=d.get("spec") if isinstance(d.get("spec"), dict) else None,
        )


class LiveManager:
    """One active recording, a ledger of all of them, and the part jobs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ledger_path = user_config.config_dir() / LEDGER_NAME
        self._records: Dict[str, LiveRecord] = {}
        self._load()
        self.active: Optional[LiveRecord] = None
        self._recorder: Optional[live_recorder.LiveRecorder] = None
        self._spec: Optional[LayoutSpec] = None
        self._language: Optional[str] = None
        self._stopped_by_user = False
        # A live the app never stopped: its recorder died with the process,
        # but the parts on disk are still good for clips.
        for rec in self._records.values():
            if rec.status in ("starting", "recording"):
                rec.status = "interrupted"
        self._save()
        self._poller = threading.Thread(target=self._poll_forever,
                                        daemon=True, name="live-poll")
        self._poller.start()

    # --- ledger -----------------------------------------------------------
    def _load(self) -> None:
        try:
            data = json.loads(self._ledger_path.read_text(encoding="utf-8"))
            for d in data if isinstance(data, list) else []:
                try:
                    rec = LiveRecord.from_ledger(d)
                    if rec.id:
                        self._records[rec.id] = rec
                except Exception:
                    continue
        except Exception:
            pass

    def _save(self) -> None:
        try:
            self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._ledger_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(
                [r.to_ledger() for r in sorted(
                    self._records.values(), key=lambda r: r.started_at,
                    reverse=True)],
                ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self._ledger_path)
        except Exception as e:  # noqa: BLE001 - the ledger must not kill a live
            print(f"[live] could not save lives.json: {e}", flush=True)

    # --- public API -------------------------------------------------------
    def status(self) -> Dict:
        with self._lock:
            return {
                "active": self.active.snapshot() if self.active else None,
                "lives": [r.snapshot() for r in sorted(
                    self._records.values(), key=lambda r: r.started_at,
                    reverse=True)][:20],
            }

    def start(self, url: str, clips_per_part: int, part_minutes: int,
              spec: LayoutSpec, language: Optional[str] = None) -> LiveRecord:
        with self._lock:
            if self.active is not None:
                raise ValueError("Ya hay un live grabándose. Páralo primero.")
            rec = LiveRecord(
                id=uuid.uuid4().hex[:10],
                url=(url or "").strip(),
                platform=live_recorder.platform_of(url),
                status="starting",
                started_at=time.time(),
                parts_dir=str(user_config.source_dir() / "lives" /
                              uuid.uuid4().hex[:10]),
                clips_per_part=max(1, min(10, int(clips_per_part or 2))),
                part_minutes=max(1, min(30, int(part_minutes or 5))),
                spec=spec.to_dict(),
            )
            # The parts folder is named after the record, not a second id.
            rec.parts_dir = str(user_config.source_dir() / "lives" / rec.id)
            self._records[rec.id] = rec
            self.active = rec
            self._spec = spec
            self._language = language
            self._stopped_by_user = False
            self._save()
        threading.Thread(target=self._resolve_and_record, args=(rec,),
                         daemon=True, name="live-start").start()
        return rec

    def stop(self, live_id: str) -> Optional[LiveRecord]:
        with self._lock:
            rec = self._records.get(live_id)
            if rec is None or self.active is not rec:
                return rec
            self._stopped_by_user = True
        if self._recorder is not None:
            self._recorder.stop()
        self._finish(rec, "stopped")
        return rec

    def delete(self, live_id: str) -> bool:
        """Drop a live from the list and delete its raw recording.

        The clips cut from it stay in the library — they are finished clips
        in their own runs, not part of the recording.
        """
        with self._lock:
            rec = self._records.get(live_id)
            if rec is None or self.active is rec:
                return False
            del self._records[live_id]
            self._save()
        try:
            shutil.rmtree(rec.parts_dir, ignore_errors=True)
        except Exception:
            pass
        return True

    def process_remaining_parts(self, live_id: str) -> int:
        """Cut clips from every recorded part that never got a job.

        For an interrupted live, and for parts that arrived after the last
        poll. Returns how many jobs were started.
        """
        with self._lock:
            rec = self._records.get(live_id)
            if rec is None:
                raise ValueError("No such live")
            spec = self._spec if self.active is rec else None
        made = 0
        parts_dir = Path(rec.parts_dir)
        if not parts_dir.is_dir():
            return 0
        done = {int(p.get("part")) for p in rec.part_jobs}
        for part in sorted(parts_dir.glob("parte-*.mp4")):
            try:
                n = int(part.stem.split("-")[1])
            except (IndexError, ValueError):
                continue
            if n in done:
                continue
            self._cut_part(rec, str(part), n, spec)
            made += 1
        return made

    # --- internals --------------------------------------------------------
    def _resolve_and_record(self, rec: LiveRecord) -> None:
        try:
            info = live_recorder.resolve_live(rec.url)
        except Exception as e:
            with self._lock:
                rec.status = "error"
                rec.error = str(e)
                rec.ended_at = time.time()
                if self.active is rec:
                    self.active = None
                self._save()
            return
        with self._lock:
            rec.title = info["title"]
            rec.platform = info["platform"]
            rec.status = "recording"
            self._save()
        recorder = live_recorder.LiveRecorder(
            rec.parts_dir, segment_seconds=rec.part_minutes * 60,
            on_part=lambda p, n: self._cut_part(rec, p, n, None),
            on_ended=lambda err: self._on_recorder_ended(rec, err))
        with self._lock:
            self._recorder = recorder
        try:
            recorder.start(info["video_url"], info.get("audio_url"))
        except Exception as e:
            with self._lock:
                rec.status = "error"
                rec.error = f"No se pudo grabar: {e}"
                rec.ended_at = time.time()
                if self.active is rec:
                    self.active = None
                self._save()

    def _cut_part(self, rec: LiveRecord, part_path: str, part_no: int,
                  spec: Optional[LayoutSpec]) -> None:
        """Send one finished part through the normal clip pipeline."""
        if not os.path.exists(part_path) or os.path.getsize(part_path) < 10000:
            return
        with self._lock:
            if any(int(p.get("part")) == part_no for p in rec.part_jobs):
                return
            use_spec = copy.deepcopy(spec or self._spec)
            if use_spec is None and rec.spec:
                # An interrupted live from an earlier session: the spec it
                # was started with is the one in the ledger.
                use_spec = LayoutSpec.from_dict(rec.spec)
            if use_spec is None:
                return
            use_spec.num_clips = rec.clips_per_part
        job = STORE.create(part_path, use_spec, language=self._language,
                           live_id=rec.id)
        with self._lock:
            rec.part_jobs.append({"part": part_no, "job_id": job.id})
            self._save()
        print(f"[live] part {part_no} → job {job.id} "
              f"({rec.clips_per_part} clips)", flush=True)

    def _on_recorder_ended(self, rec: LiveRecord, err: Optional[str]) -> None:
        with self._lock:
            if self.active is not rec:
                return
            if self._stopped_by_user:
                return  # stop() finishes it
        if err:
            self._finish(rec, "interrupted", error=f"La grabación se cortó: {err}")
        else:
            self._finish(rec, "ended")

    def _finish(self, rec: LiveRecord, status: str,
                error: Optional[str] = None) -> None:
        with self._lock:
            rec.status = status
            rec.ended_at = time.time()
            rec.error = error
            if self.active is rec:
                self.active = None
            self._recorder = None
            self._spec = None
            self._save()
        print(f"[live] {rec.title or rec.url} → {status}", flush=True)

    def _poll_forever(self) -> None:
        while True:
            time.sleep(5.0)
            try:
                self._refresh()
            except Exception as e:  # noqa: BLE001 - the poller must not die
                print(f"[live] poll: {e}", flush=True)

    def _refresh(self) -> None:
        """Keep each live's clip count and part-job states current."""
        with self._lock:
            records = list(self._records.values())
        changed = False
        for rec in records:
            total = 0
            for p in rec.part_jobs:
                snap = None
                job = STORE.get(p["job_id"])
                if job is not None:
                    snap = job.snapshot()
                if snap is None:
                    continue
                total += sum(1 for c in snap["clips"] if c.get("url"))
                p["status"] = snap["status"]
                p["progress"] = snap["progress"]
                p["message"] = snap["message"]
                p["clips"] = sum(1 for c in snap["clips"] if c.get("url"))
            if total != rec.clips_count:
                rec.clips_count = total
                changed = True
        if changed:
            with self._lock:
                self._save()


MANAGER = LiveManager()
