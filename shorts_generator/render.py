"""One entry point for every layout.

The repo grew two renderers with different shapes — the face-tracking crop in
local/clipper.py and the stacked webcam layout in local/gaming_layout.py — plus
a plain centre crop that had no home. This dispatches a LayoutSpec to whichever
one it asks for, so callers (CLI, web UI) never branch on layout themselves.
"""
import os
import subprocess
from typing import Callable, Dict, List, Optional

from . import accel, proc
from .layout_spec import LayoutSpec

# Loudness target for every export. -14 LUFS is what YouTube, TikTok and
# Instagram all normalise toward, so hitting it means nothing is turned down
# on upload and nothing plays quiet next to whatever the feed shows first.
LOUDNESS_FILTER = "loudnorm=I=-14:TP=-1.5:LRA=11"


def _render_center_clip(source_path: str, start: float, end: float, out_path: str,
                        out_w: int, out_h: int, frame: Optional[Dict] = None) -> Dict:
    """Widest centre crop at the target ratio, in one ffmpeg pass.

    No face detection at all — for footage where the webcam is irrelevant or
    the user explicitly asked for gameplay only. `frame` is a position chosen
    by hand, which replaces the centre.
    """
    from . import framing
    from .local.gaming_layout import _probe_dimensions

    src_w, src_h = _probe_dimensions(source_path)
    crop_w, crop_h = framing.picture_window(src_w, src_h, out_w, out_h)
    x, y = framing.place(frame, src_w, src_h, crop_w, crop_h,
                         ((src_w - crop_w) // 2, (src_h - crop_h) // 2))

    filt = (
        f"[0:v]crop={crop_w}:{crop_h}:{x}:{y},"
        f"scale={out_w}:{out_h}:flags=lanczos,setsar=1[v]"
    )
    accel.run_encode(lambda enc: [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-i", source_path, "-t", f"{end - start:.3f}",
        "-filter_complex", filt,
        "-map", "[v]", "-map", "0:a:0?",
        # Every Short lands in a feed between two videos mastered by someone
        # else. A clip that plays quieter than what came before it reads as
        # lower production value before a word of it is heard, so the output
        # is normalised to the same target the platforms mix to.
        "-af", LOUDNESS_FILTER,
        *enc,
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        out_path,
    ], what="ffmpeg (centre crop render)", crf=23)
    return {"crop": {"x": x, "y": y, "w": crop_w, "h": crop_h},
            "panel": framing.PICTURE, "manual": bool(framing.normalise(frame)),
            "x": framing.fraction(x, src_w, crop_w), "y": framing.fraction(y, src_h, crop_h)}


def _render_center_highlights(source_path: str, highlights: List[Dict], out_dir: str,
                              out_w: int, out_h: int, name_prefix: str = "short") -> List[Dict]:
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict] = []
    for i, h in enumerate(highlights, 1):
        out_path = os.path.join(out_dir, f"{name_prefix}_{i:02d}.mp4")
        print(f"[center] {i}/{len(highlights)}: {h.get('title', '(untitled)')}", flush=True)
        try:
            info = _render_center_clip(
                source_path, float(h["start_time"]), float(h["end_time"]),
                out_path, out_w, out_h, frame=h.get("frame"),
            )
            results.append({**h, "clip_url": out_path, "layout": info})
        except Exception as e:
            print(f"[center] {i} failed: {e}", flush=True)
            results.append({**h, "clip_url": None, "error": str(e)})
    return results


def render_highlights(
    source_path: str,
    highlights: List[Dict],
    spec: LayoutSpec,
    out_dir: Optional[str] = None,
    name_prefix: str = "short",
    language: Optional[str] = None,
    kind: Optional[str] = None,
    llm_fn: Optional[Callable[[str], str]] = None,
) -> List[Dict]:
    """Render `highlights` from `source_path` according to `spec`.

    Returns the highlights with `clip_url` set (or `error` on failure) —
    the same shape every renderer in this repo already produces.

    `language` is the spoken language, for the captions; `kind` is what sort
    of video this turned out to be, which decides how hard pauses are cut;
    `llm_fn` is only used to place B-roll, and B-roll is skipped without it.
    """
    from .config import LOCAL_OUTPUT_DIR
    from .layout_spec import pick_output_size
    from .local.gaming_layout import _probe_dimensions

    out_dir = out_dir or LOCAL_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    out_w, out_h = spec.width, spec.height
    if spec.match_source_quality:
        try:
            src_w, src_h = _probe_dimensions(source_path)
            out_w, out_h = pick_output_size(src_w, src_h, spec.aspect_ratio, spec.layout)
            if (out_w, out_h) != (spec.width, spec.height):
                print(f"[render] source is {src_w}x{src_h} — rendering at "
                      f"{out_w}x{out_h} instead of {spec.width}x{spec.height}", flush=True)
        except Exception as e:
            print(f"[render] could not probe source ({e}); using {out_w}x{out_h}", flush=True)

    print(f"[render] {spec.layout} · {spec.describe()} · {out_w}x{out_h}", flush=True)

    results = _render_by_layout(source_path, highlights, spec, out_dir,
                                out_w, out_h, name_prefix)
    _edit(results, spec, language, kind, llm_fn)
    if spec.hook_replay:
        _add_hook_openings(results)
    if getattr(spec, "outro", True):
        _append_outro(results, out_w, out_h)
    _write_subtitles(results)
    return results


# ClipZoneRD's animated outro, appended to the end of every rendered clip.
OUTRO_FILE = "outro-animated-v5.mp4"


def _outro_path() -> Optional[str]:
    """The bundled outro file, from the build or the repo, or None."""
    from . import bundled
    d = bundled.asset_dir("outro")
    if d is not None:
        p = d / OUTRO_FILE
        if p.is_file():
            return str(p)
    p = os.path.join(os.path.dirname(__file__), "..", "assets", "outro", OUTRO_FILE)
    p = os.path.normpath(p)
    return p if os.path.isfile(p) else None


def _probe_clip(path: str, out_w: int = 720, out_h: int = 1280) -> tuple:
    """(width, height, fps, has_audio) for a rendered clip, via ffprobe.

    Never raises: on any failure it reports the size the render was asked
    for, 30fps, with audio.
    """
    width, height, fps, has_audio = out_w, out_h, 30.0, True
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,avg_frame_rate",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=15).stdout.strip()
        parts = out.split(",")
        if len(parts) >= 2:
            width, height = int(parts[0]), int(parts[1])
        if len(parts) >= 3 and "/" in parts[2]:
            num, den = parts[2].split("/", 1)
            fps = float(num) / float(den) if float(den) else 30.0
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=15).stdout.strip()
        has_audio = out == "audio"
    except Exception:
        pass
    return width, height, fps, has_audio


def _append_outro(results: List[Dict], out_w: int, out_h: int) -> None:
    """Append the ClipZoneRD outro to every rendered clip, in place.

    The outro is scaled to the clip's own size (pillarboxed when the clip
    is not 9:16) and concatenated after it, audio and all — the clip's own
    video is never re-scaled, so joining cannot cost it quality. A clip that
    cannot be joined keeps its plain cut and says why in the log — the
    outro is never worth losing a clip over.
    """
    outro = _outro_path()
    if outro is None:
        print("[outro] bundled outro not found — clips ship without it", flush=True)
        return
    for r in results:
        path = r.get("clip_url")
        if not path or not os.path.exists(path):
            continue
        clip_w, clip_h, fps, has_audio = _probe_clip(path, out_w, out_h)
        fps_s = f"{fps:.2f}".rstrip("0").rstrip(".")
        audio0 = ("[0:a]aresample=48000,aformat=channel_layouts=stereo[a0]"
                  if has_audio else
                  "anullsrc=channel_layout=stereo:sample_rate=48000[a0]")
        filt = (
            "[0:v]setsar=1,format=yuv420p[v0];"
            f"[1:v]scale={clip_w}:{clip_h}:force_original_aspect_ratio=decrease,"
            f"pad={clip_w}:{clip_h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps_s},"
            "format=yuv420p[v1];"
            f"{audio0};"
            "[1:a]aresample=48000,aformat=channel_layouts=stereo[a1];"
            "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]"
        )
        tmp = path + ".outro.mp4"
        try:
            accel.run_encode(lambda enc: [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", path, "-i", outro,
                "-filter_complex", filt,
                "-map", "[v]", "-map", "[a]",
                *enc,
                "-c:a", "aac", "-b:a", "160k",
                "-movflags", "+faststart",
                tmp,
            ], what="ffmpeg (append outro)", crf=23)
            os.replace(tmp, path)
            r["outro"] = True
        except Exception as e:  # noqa: BLE001 - a plain clip beats no clip
            first = (str(e).strip().splitlines() or [e.__class__.__name__])[-1][:200]
            print(f"[outro] could not append to {os.path.basename(path)} "
                  f"({first}) — keeping the plain cut", flush=True)
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass


def _srt_timestamp(seconds: float) -> str:
    """Seconds -> an SRT timestamp (HH:MM:SS,mmm)."""
    ms = max(0, int(round(float(seconds) * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    sec, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def _write_subtitles(results: List[Dict]) -> None:
    """Write a separate .srt subtitle file beside every rendered clip.

    The words come from the edit pass, on the clip's final timeline, shifted
    by the cold open when one was added. The .srt is the default subtitle
    delivery — burned-in captions are an option, not the default — so it is
    written even when burned-in captions are off.
    """
    for r in results:
        path = r.get("clip_url")
        words = r.get("subtitle_words")
        if not path or not os.path.exists(path) or not words:
            continue
        shift = float(r.get("hook_replay_seconds") or 0.0)
        # Group the words into cues: up to 5 words, 2.5s, ~84 characters.
        cues = []
        cur = []
        for w in words:
            try:
                start, end, text = float(w["start"]), float(w["end"]), str(w["word"])
            except (KeyError, TypeError, ValueError):
                continue
            if cur and (len(cur) >= 5
                        or end - float(cur[0]["start"]) > 2.5
                        or sum(len(c["word"]) for c in cur) + len(text) > 84):
                cues.append(cur)
                cur = []
            cur.append({"start": start, "end": end, "word": text})
        if cur:
            cues.append(cur)
        if not cues:
            continue
        root, _ = os.path.splitext(path)
        srt_path = root + ".srt"
        try:
            with open(srt_path, "w", encoding="utf-8") as f:
                n = 0
                for cue in cues:
                    start = float(cue[0]["start"]) + shift
                    end = float(cue[-1]["end"]) + shift
                    text = " ".join(c["word"] for c in cue).strip()
                    if not text or end <= start:
                        continue
                    n += 1
                    f.write(f"{n}\n{_srt_timestamp(start)} --> "
                            f"{_srt_timestamp(end)}\n{text}\n\n")
            r["subtitles"] = os.path.basename(srt_path)
        except OSError as e:
            print(f"[subtitles] could not write {srt_path} ({e})", flush=True)


def _edit(results: List[Dict], spec: LayoutSpec, language: Optional[str],
          kind: Optional[str], llm_fn) -> None:
    """Captions, jump cuts, punch-ins, emoji and B-roll -- see autoedit.

    After the layout, so all three renderers get it from one place, and before
    the cold open, so the replayed moment carries its captions with it.
    """
    from .autoedit import options_from_spec, polish

    opts = options_from_spec(spec, kind)
    if not opts.anything:
        return
    polish(results, opts, language=language, llm_fn=llm_fn)


def _add_hook_openings(results: List[Dict]) -> None:
    """Give each rendered clip a cold open on its own loudest moment.

    Done after rendering rather than inside each renderer, so all three
    layouts get it from one place and none of them has to know it exists. It
    no-ops on any clip whose hook is already at the front.
    """
    from .hook_open import apply

    for r in results:
        path = r.get("clip_url")
        if not path or not os.path.exists(path):
            continue
        target = r
        # Cutting pauses moved the peak earlier in the clip. The cold open
        # has to point at where it is now, not where it was in the source.
        offset = r.pop("hook_peak_offset", None)
        if offset is not None and r.get("start_time") is not None:
            target = {**r, "hook_peak": float(r["start_time"]) + float(offset)}
        added = apply(path, target)
        if added:
            r["hook_replay_seconds"] = round(added, 2)


def _render_by_layout(source_path: str, highlights: List[Dict], spec: LayoutSpec,
                      out_dir: str, out_w: int, out_h: int,
                      name_prefix: str) -> List[Dict]:
    if spec.layout == "stacked":
        from .local.gaming_layout import render_stacked_highlights
        return render_stacked_highlights(
            source_path, highlights, out_dir=out_dir,
            corner=spec.webcam_corner,
            out_w=out_w, out_h=out_h,
            cam_panel_fraction=spec.cam_panel_fraction,
            face_context_multiple=spec.face_zoom,
            name_prefix=name_prefix,
        )

    if spec.layout == "facetrack":
        from .local.clipper import crop_highlights_local
        return crop_highlights_local(
            source_path, highlights,
            aspect_ratio=spec.aspect_ratio,
            out_dir=out_dir,
            target_resolution=f"{out_w}x{out_h}",
            name_prefix=name_prefix,
        )

    return _render_center_highlights(
        source_path, highlights, out_dir, out_w, out_h, name_prefix=name_prefix
    )
