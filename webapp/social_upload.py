"""Sharing a finished clip to Facebook, Instagram and TikTok, one click.

Only the official APIs, and only what they actually allow. The shortcut --
drive a social app's upload page with a scripted browser -- is against every
one of these platforms' terms, breaks whenever they move a button, and puts
the account at risk. An API upload is the thing each platform built for
exactly this.

How the pieces fit (the same shape as webapp/youtube_upload.py):

  Signing in.  The user clicks Connect, their real browser opens the
  platform's own consent page, and the platform redirects back to this app's
  local server with a one-time code. The app never sees a password. What it
  keeps is a token, in the per-user config folder beside the API keys.

  The client.  Each platform identifies an app by an OAuth client ID (and a
  secret). There is no built-in one: the user pastes their own, from an app
  they create in the platform's developer portal -- Meta for Developers for
  Facebook/Instagram, TikTok for Developers for TikTok. The Settings panel
  says exactly where to make one.

  What each platform actually allows, and what it does not:

  Facebook (Pages).  A Meta app, scopes pages_manage_posts +
  pages_read_engagement + pages_show_list. The Page token never expires, so
  signing in once is enough. Reels go up in three phases (start, bytes to
  rupload.facebook.com, finish as PUBLISHED). A Reel is 3-90 seconds, 9:16.
  No App Review is needed to post to a Page she owns.

  Instagram.  The Instagram API with Instagram Login (graph.instagram.com),
  scopes instagram_business_basic + instagram_business_content_publish. The
  account must be Professional (Business or Creator) -- a personal account
  cannot publish through the API. Reels go up as a resumable upload and are
  then published. About 90 seconds through the API, up to 300 MB.

  TikTok.  The Content Posting API. Inbox drafts (video.upload) work without
  any review: the clip lands as a draft in her TikTok app and she publishes
  it there, which is also how her approve-before-publish rule is kept.
  Direct posting to the public feed (video.publish) needs TikTok to audit the
  app first -- a fully developed, externally facing website, a Privacy Policy
  and Terms, and a demo video -- and an app that only runs on her own PC does
  not pass that audit. So this app drafts; it does not publish to TikTok
  directly, and it says so where the button is.

Uploading happens only when she presses Compartir, with the title,
description and hashtags the app already wrote for the clip, and those words
are never trimmed or rewritten on the way. A share is recorded on the clip
(shared_to), so the monthly panel can count it.
"""
import base64
import hashlib
import json
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlencode

import requests

from shorts_generator import user_config

HTTP_TIMEOUT = 60
CHUNK = 8 * 1024 * 1024

_lock = threading.RLock()
_pending: Dict[str, Dict] = {}


class NotConnected(Exception):
    """No usable sign-in for that platform."""


class ShareError(Exception):
    pass


# ------------------------------------------------------------------ config

PROVIDERS = ("facebook", "instagram", "tiktok")

# What each platform needs from its developer portal, in the words the
# Settings panel shows next to the paste box.
CLIENT_HELP = {
    "facebook": ("A Meta app (developers.facebook.com → My Apps → Create App → "
                 "Other → Business). Its App ID and App secret. Add Facebook Login "
                 "and put http://127.0.0.1:PORT/api/share/facebook/callback in Valid "
                 "OAuth Redirect URIs (the port is the one this app runs on)."),
    "instagram": ("An Instagram API with Instagram Login app "
                  "(developers.facebook.com → My Apps → Create App → Other → "
                  "Instagram API with Instagram Login). Its Instagram App ID and "
                  "App Secret, and http://127.0.0.1:PORT/api/share/instagram/callback "
                  "in Valid OAuth Redirect URIs. The account must be Professional."),
    "tiktok": ("A TikTok developer app (developers.tiktok.com → Manage apps → "
               "Create an app). Its Client key and Client secret, with the Login Kit "
               "and Content Posting API products added, and "
               "http://127.0.0.1:PORT/api/share/tiktok/callback as a redirect URI. "
               "No audit is needed for drafts."),
}

SCOPES = {
    "facebook": ["pages_manage_posts", "pages_read_engagement", "pages_show_list"],
    "instagram": ["instagram_business_basic", "instagram_business_content_publish"],
    "tiktok": ["user.info.basic", "video.upload"],
}

AUTH_URL = {
    "facebook": "https://www.facebook.com/v21.0/dialog/oauth",
    "instagram": "https://www.instagram.com/oauth/authorize",
    "tiktok": "https://www.tiktok.com/v2/auth/authorize/",
}

TOKEN_URL = {
    "facebook": "https://graph.facebook.com/v21.0/oauth/access_token",
    "instagram": "https://api.instagram.com/oauth/access_token",
    "tiktok": "https://open.tiktokapis.com/v2/oauth/token/",
}

GRAPH = {
    "facebook": "https://graph.facebook.com/v21.0",
    "instagram": "https://graph.instagram.com/v21.0",
    "tiktok": "https://open.tiktokapis.com/v2",
}

# TikTok's caption limit, and the one Facebook and Instagram share.
CAPTION_LIMIT = 2200


def _client_path(provider: str) -> Path:
    return Path(user_config.config_dir()) / f"social_{provider}_client.json"


def _token_path(provider: str) -> Path:
    return Path(user_config.config_dir()) / f"social_{provider}_token.json"


def _read_json(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def client(provider: str) -> Optional[Dict]:
    d = _read_json(_client_path(provider))
    if not d or not d.get("id") or not d.get("secret"):
        return None
    return {"id": str(d["id"]).strip(), "secret": str(d["secret"]).strip()}


def set_client(provider: str, client_id: str, secret: str) -> Dict:
    if provider not in PROVIDERS:
        raise ValueError("Unknown platform")
    if not (client_id or "").strip() or not (secret or "").strip():
        raise ValueError("Paste the app ID and the secret.")
    _write_json(_client_path(provider),
                {"id": client_id.strip(), "secret": secret.strip()})
    return status(provider)


def _load_token(provider: str) -> Optional[Dict]:
    return _read_json(_token_path(provider))


def _save_token(provider: str, tok: Dict) -> None:
    _write_json(_token_path(provider), tok)


def _forget_token(provider: str) -> None:
    try:
        _token_path(provider).unlink()
    except OSError:
        pass


def disconnect(provider: str) -> Dict:
    if provider not in PROVIDERS:
        raise ValueError("Unknown platform")
    _forget_token(provider)
    return status(provider)


# ------------------------------------------------------------------ status

def status(provider: str) -> Dict:
    if provider not in PROVIDERS:
        raise ValueError("Unknown platform")
    tok = _load_token(provider)
    out = {"provider": provider, "connected": False,
           "has_client": bool(client(provider)), "account": None,
           "needs": None, "connect_state": None}
    if not tok:
        out["needs"] = ("Connect your account first." if out["has_client"]
                        else "Add your app ID and secret first.")
        return out
    out["connected"] = True
    out["account"] = tok.get("account")
    out["expires_at"] = tok.get("expires_at")
    return out


def all_status() -> Dict:
    return {p: status(p) for p in PROVIDERS}


# ------------------------------------------------------------------ connect

def begin_connect(provider: str, redirect_uri: str) -> str:
    """The platform's consent page to send the user to."""
    c = client(provider)
    if not c:
        raise NotConnected("No app is set up yet. Add the App ID and secret first.")
    state = secrets.token_urlsafe(24)
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    with _lock:
        now = time.time()
        for k in [k for k, v in _pending.items() if now - v["started"] > 900]:
            _pending.pop(k, None)
        _pending[state] = {"provider": provider, "verifier": verifier,
                           "redirect_uri": redirect_uri, "client": c,
                           "started": now}
    params = {"client_id": c["id"], "redirect_uri": redirect_uri,
              "state": state}
    if provider == "tiktok":
        params.update({"response_type": "code", "scope": " ".join(SCOPES[provider]),
                       "code_challenge": challenge,
                       "code_challenge_method": "S256"})
    elif provider == "instagram":
        params.update({"response_type": "code", "scope": " ".join(SCOPES[provider])})
    else:  # facebook
        params.update({"response_type": "code", "scope": ",".join(SCOPES[provider])})
    return AUTH_URL[provider] + "?" + urlencode(params)


def finish_connect(provider: str, state: str, code: str, error: str) -> Optional[str]:
    """Finish a sign-in from the callback. Returns the problem, or None."""
    if error:
        return f"The platform said no: {error}"
    with _lock:
        pending = _pending.pop(state, None)
    if not pending or pending.get("provider") != provider:
        return "That sign-in was never started here, or it expired."
    c = pending["client"]
    redirect_uri = pending["redirect_uri"]
    try:
        if provider == "facebook":
            tok = _exchange_facebook(c, code, redirect_uri)
        elif provider == "instagram":
            tok = _exchange_instagram(c, code, redirect_uri)
        else:
            tok = _exchange_tiktok(c, code, redirect_uri, pending["verifier"])
    except ShareError as e:
        return str(e)
    _save_token(provider, tok)
    return None


def _exchange_facebook(c: Dict, code: str, redirect_uri: str) -> Dict:
    r = requests.get(TOKEN_URL["facebook"], params={
        "client_id": c["id"], "client_secret": c["secret"],
        "redirect_uri": redirect_uri, "code": code}, timeout=HTTP_TIMEOUT)
    data = _api_result(r, "Facebook")
    short = data.get("access_token")
    if not short:
        raise ShareError("Facebook did not hand back a token.")
    # A long-lived user token (60 days), then the Page token, which never
    # expires -- that is what uploads Reels, and it is what is kept.
    r = requests.get(TOKEN_URL["facebook"], params={
        "grant_type": "fb_exchange_token", "client_id": c["id"],
        "client_secret": c["secret"], "fb_exchange_token": short},
        timeout=HTTP_TIMEOUT)
    long_tok = _api_result(r, "Facebook").get("access_token") or short
    r = requests.get(f"{GRAPH['facebook']}/me/accounts",
                     params={"access_token": long_tok}, timeout=HTTP_TIMEOUT)
    pages = _api_result(r, "Facebook").get("data") or []
    if not pages:
        raise ShareError("No Facebook Page was found on that account. "
                         "The app posts to a Page, not a personal profile.")
    page = pages[0]
    return {"access_token": page.get("access_token"),
            "user_token": long_tok,
            "account": {"type": "page", "id": page.get("id"),
                        "name": page.get("name")},
            "expires_at": None}


def _exchange_instagram(c: Dict, code: str, redirect_uri: str) -> Dict:
    r = requests.post(TOKEN_URL["instagram"], data={
        "client_id": c["id"], "client_secret": c["secret"],
        "grant_type": "authorization_code", "redirect_uri": redirect_uri,
        "code": code}, timeout=HTTP_TIMEOUT)
    data = _api_result(r, "Instagram")
    short = data.get("access_token")
    user_id = data.get("user_id")
    if not short or not user_id:
        raise ShareError("Instagram did not hand back a token.")
    r = requests.get(f"{GRAPH['instagram']}/access_token", params={
        "grant_type": "ig_exchange_token", "client_secret": c["secret"],
        "access_token": short}, timeout=HTTP_TIMEOUT)
    long_tok = _api_result(r, "Instagram").get("access_token") or short
    r = requests.get(f"{GRAPH['instagram']}/me",
                     params={"fields": "id,username", "access_token": long_tok},
                     timeout=HTTP_TIMEOUT)
    me = _api_result(r, "Instagram")
    return {"access_token": long_tok,
            "account": {"type": "instagram", "id": me.get("id") or user_id,
                        "name": me.get("username") or ""},
            "expires_at": time.time() + 60 * 86400}


def _exchange_tiktok(c: Dict, code: str, redirect_uri: str, verifier: str) -> Dict:
    r = requests.post(TOKEN_URL["tiktok"], data={
        "client_key": c["id"], "client_secret": c["secret"],
        "code": code, "grant_type": "authorization_code",
        "redirect_uri": redirect_uri, "code_verifier": verifier},
        timeout=HTTP_TIMEOUT)
    data = _api_result(r, "TikTok")
    tok = data.get("data") or {}
    access = tok.get("access_token")
    if not access:
        raise ShareError("TikTok did not hand back a token.")
    r = requests.get(f"{GRAPH['tiktok']}/user/info/",
                     params={"fields": "open_id,display_name"},
                     headers={"Authorization": f"Bearer {access}"},
                     timeout=HTTP_TIMEOUT)
    me = (_api_result(r, "TikTok").get("data") or {}).get("user") or {}
    return {"access_token": access,
            "refresh_token": tok.get("refresh_token"),
            "open_id": me.get("open_id") or tok.get("open_id"),
            "account": {"type": "tiktok", "id": me.get("open_id") or "",
                        "name": me.get("display_name") or ""},
            "expires_at": time.time() + int(tok.get("expires_in") or 7200),
            "refresh_expires_at": time.time() + int(
                tok.get("refresh_expires_in") or 180 * 86400)}


def _api_result(r: requests.Response, who: str) -> Dict:
    try:
        data = r.json()
    except Exception:
        data = {}
    if r.status_code >= 400 or (isinstance(data, dict) and data.get("error")):
        err = data.get("error") or {}
        msg = (err.get("message") if isinstance(err, dict) else str(err)) or data
        raise ShareError(f"{who}: {msg}")
    return data if isinstance(data, dict) else {}


# ------------------------------------------------------------------ tokens

def _facebook_page_token() -> Dict:
    tok = _load_token("facebook")
    if not tok or not tok.get("access_token") or not tok.get("account"):
        raise NotConnected("Facebook is not connected.")
    return tok


def _instagram_token() -> str:
    tok = _load_token("instagram")
    if not tok or not tok.get("access_token"):
        raise NotConnected("Instagram is not connected.")
    return str(tok["access_token"])


def _tiktok_token() -> str:
    tok = _load_token("tiktok")
    if not tok:
        raise NotConnected("TikTok is not connected.")
    if tok.get("expires_at") and time.time() > float(tok["expires_at"]) - 60:
        tok = _refresh_tiktok(tok)
    return str(tok["access_token"])


def _refresh_tiktok(tok: Dict) -> Dict:
    c = client("tiktok")
    if not c or not tok.get("refresh_token"):
        raise NotConnected("TikTok's sign-in expired. Connect it again.")
    r = requests.post(TOKEN_URL["tiktok"], data={
        "client_key": c["id"], "client_secret": c["secret"],
        "grant_type": "refresh_token", "refresh_token": tok["refresh_token"]},
        timeout=HTTP_TIMEOUT)
    data = (_api_result(r, "TikTok").get("data") or {})
    if not data.get("access_token"):
        raise NotConnected("TikTok would not refresh the sign-in. Connect it again.")
    # The refresh token rotates on every refresh: keep the new one.
    tok = dict(tok)
    tok["access_token"] = data["access_token"]
    tok["refresh_token"] = data.get("refresh_token") or tok.get("refresh_token")
    tok["expires_at"] = time.time() + int(data.get("expires_in") or 7200)
    _save_token("tiktok", tok)
    return tok


# ------------------------------------------------------------------ captions

def build_caption(title: str, description: str, tags: List[str]) -> str:
    """The caption that goes with a clip: title, description, hashtags."""
    parts = []
    if (title or "").strip():
        parts.append(title.strip())
    if (description or "").strip():
        parts.append(description.strip())
    tags = [t for t in (tags or []) if str(t).strip()]
    if tags:
        parts.append(" ".join(f"#{str(t).strip().lstrip('#')}" for t in tags))
    caption = "\n\n".join(parts)
    if len(caption) > CAPTION_LIMIT:
        caption = caption[:CAPTION_LIMIT].rsplit(" ", 1)[0].rstrip()
    return caption


# ------------------------------------------------------------------ uploads

def _read_clip(path: Path) -> bytes:
    data = path.read_bytes()
    if not data:
        raise ShareError("That clip's file is empty.")
    return data


def upload_facebook(path: Path, title: str, description: str,
                    tags: List[str]) -> Dict:
    """A Reel on the connected Facebook Page, published at once."""
    tok = _facebook_page_token()
    page = tok["account"]
    caption = build_caption(title, description, tags)
    size = path.stat().st_size
    # Phase 1: start.
    r = requests.post(f"{GRAPH['facebook']}/{page['id']}/video_reels", data={
        "upload_phase": "start", "access_token": tok["access_token"]},
        timeout=HTTP_TIMEOUT)
    started = _api_result(r, "Facebook")
    video_id, upload_url = started.get("video_id"), started.get("upload_url")
    if not video_id or not upload_url:
        raise ShareError("Facebook did not start the upload.")
    # Phase 2: the bytes.
    data = _read_clip(path)
    r = requests.post(upload_url,
                      headers={"Authorization": f"OAuth {tok['access_token']}",
                               "file_offset": "0"},
                      data=data, timeout=300)
    _api_result(r, "Facebook")
    # Phase 3: finish, published.
    r = requests.post(f"{GRAPH['facebook']}/{page['id']}/video_reels", data={
        "upload_phase": "finish", "video_id": video_id,
        "video_state": "PUBLISHED", "description": caption,
        "access_token": tok["access_token"]}, timeout=HTTP_TIMEOUT)
    done = _api_result(r, "Facebook")
    if not done.get("success", True):
        raise ShareError("Facebook did not finish the Reel.")
    return {"platform": "facebook", "video_id": video_id,
            "url": f"https://www.facebook.com/{page['id']}/videos/{video_id}",
            "size": size}


def upload_instagram(path: Path, title: str, description: str,
                     tags: List[str]) -> Dict:
    """A Reel on the connected Instagram account, published at once."""
    token = _instagram_token()
    tok = _load_token("instagram") or {}
    user_id = (tok.get("account") or {}).get("id")
    if not user_id:
        raise NotConnected("Instagram is not connected.")
    caption = build_caption(title, description, tags)
    data = _read_clip(path)
    # 1. A resumable upload: the container and the URL to put the bytes on.
    r = requests.post(f"{GRAPH['instagram']}/{user_id}/media", data={
        "upload_type": "resumable", "media_type": "REELS",
        "caption": caption, "access_token": token}, timeout=HTTP_TIMEOUT)
    created = _api_result(r, "Instagram")
    uri, container = created.get("uri"), created.get("id")
    if not uri or not container:
        raise ShareError("Instagram did not start the upload.")
    # 2. The bytes.
    r = requests.put(uri, headers={"Authorization": f"Bearer {token}",
                                   "Content-Type": "application/octet-stream",
                                   "Offset": "0"},
                     data=data, timeout=300)
    if r.status_code >= 400:
        raise ShareError(f"Instagram: the upload failed ({r.status_code}).")
    # 3. Publish the container.
    r = requests.post(f"{GRAPH['instagram']}/{user_id}/media_publish", data={
        "creation_id": container, "access_token": token},
        timeout=HTTP_TIMEOUT)
    done = _api_result(r, "Instagram")
    media_id = done.get("id")
    if not media_id:
        raise ShareError("Instagram did not publish the Reel.")
    return {"platform": "instagram", "video_id": media_id,
            "url": f"https://www.instagram.com/reel/{media_id}/",
            "size": len(data)}


def upload_tiktok_draft(path: Path, title: str, description: str,
                        tags: List[str]) -> Dict:
    """A draft in her TikTok app's inbox, for her to publish there.

    Direct posting to the public feed needs TikTok to audit the app, and an
    app that only runs on her own PC does not pass that audit -- so this is
    deliberately a draft, not a publish. The draft carries the title,
    description and hashtags, and she taps Publish in the TikTok app.
    """
    token = _tiktok_token()
    caption = build_caption(title, description, tags)
    data = _read_clip(path)
    total = (len(data) + CHUNK - 1) // CHUNK
    r = requests.post(f"{GRAPH['tiktok']}/post/publish/inbox/video/init/",
                      headers={"Authorization": f"Bearer {token}",
                               "Content-Type": "application/json; charset=UTF-8"},
                      json={"post_info": {
                          "title": caption,
                          "privacy_level": "SELF_ONLY",
                          "disable_duet": False, "disable_comment": False,
                          "disable_stitch": False,
                          "video_cover_timestamp_ms": 1000},
                          "source_info": {
                              "source": "FILE_UPLOAD",
                              "video_size": len(data),
                              "chunk_size": CHUNK,
                              "total_chunk_count": total}},
                      timeout=HTTP_TIMEOUT)
    init = (_api_result(r, "TikTok").get("data") or {})
    publish_id, upload_url = init.get("publish_id"), init.get("upload_url")
    if not publish_id or not upload_url:
        raise ShareError("TikTok did not start the upload.")
    for n in range(total):
        chunk = data[n * CHUNK:(n + 1) * CHUNK]
        r = requests.put(upload_url,
                         headers={"Content-Type": "video/mp4",
                                  "Content-Range": f"bytes {n * CHUNK}-"
                                  f"{n * CHUNK + len(chunk) - 1}/{len(data)}"},
                         data=chunk, timeout=300)
        if r.status_code >= 400:
            raise ShareError(f"TikTok: chunk {n + 1} failed ({r.status_code}).")
    return {"platform": "tiktok", "video_id": publish_id, "draft": True,
            "url": None,
            "note": "Draft in your TikTok app's inbox — open TikTok and publish it there.",
            "size": len(data)}


UPLOADERS = {"facebook": upload_facebook, "instagram": upload_instagram,
             "tiktok": upload_tiktok_draft}
