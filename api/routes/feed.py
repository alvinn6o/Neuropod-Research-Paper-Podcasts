from __future__ import annotations

import io
import uuid
from datetime import datetime, timezone
from email.utils import format_datetime
from xml.sax.saxutils import escape

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import RedirectResponse, StreamingResponse

from .. import audio_store, store_db
from ..db import get_user_by_slug

router = APIRouter(prefix="/feed", tags=["feed"])


def _episode_to_item(request: Request, episode: dict, user_slug: str) -> str:
    base = str(request.base_url).rstrip("/")
    audio_key = episode.get("audio_key") or ""
    audio_url = ""
    if audio_key:
        audio_url = audio_store.url_for(audio_key) or (
            f"{base}/feed/{user_slug}/episodes/{episode['id']}/audio"
        )
    page_url = f"{base}/episodes/{episode['id']}"
    enclosure = ""
    if audio_url:
        enclosure = (
            f'<enclosure url="{escape(audio_url)}" '
            f'type="{escape(episode.get("audio_mime", "audio/mpeg"))}" length="0" />'
        )

    created = episode.get("created_at") or datetime.now(timezone.utc).isoformat()
    try:
        pub_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
    except Exception:
        pub_dt = datetime.now(timezone.utc)
    pub_date = format_datetime(pub_dt)

    return f"""
        <item>
          <title>{escape(episode['title'])}</title>
          <link>{page_url}</link>
          <guid isPermaLink="false">{episode['id']}</guid>
          <description>{escape(episode['description'])}</description>
          <pubDate>{pub_date}</pubDate>
          {enclosure}
        </item>""".strip()


@router.get("/{user_slug}")
def get_feed(user_slug: str, request: Request) -> Response:
    user = get_user_by_slug(user_slug)
    if not user:
        raise HTTPException(status_code=404, detail="feed not found")

    episodes = store_db.list_episodes(uuid.UUID(str(user["id"])), limit=100)
    base = str(request.base_url).rstrip("/")
    title = f"Neuropod - {escape(user.get('display_name') or user_slug)}"
    items = "\n".join(_episode_to_item(request, ep, user_slug) for ep in episodes)

    body = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
  <channel>
    <title>{title}</title>
    <link>{base}</link>
    <description>Research papers, distilled into audio.</description>
    <language>en-us</language>
    {items}
  </channel>
</rss>"""
    return Response(content=body, media_type="application/rss+xml; charset=utf-8")


@router.get("/{user_slug}/episodes/{episode_id}/audio")
def get_feed_audio(user_slug: str, episode_id: str):
    try:
        eid = uuid.UUID(episode_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid episode id")

    episode = store_db.get_episode_by_slug(user_slug, eid)
    if not episode:
        raise HTTPException(status_code=404, detail="episode not found")
    audio_key = episode.get("audio_key")
    if not audio_key:
        raise HTTPException(status_code=404, detail="audio not generated")

    presigned = audio_store.url_for(audio_key)
    if presigned:
        return RedirectResponse(presigned, status_code=302)

    fetched = audio_store.fetch(audio_key)
    if not fetched:
        raise HTTPException(status_code=404, detail="audio missing from storage")
    audio_bytes, mime = fetched
    return StreamingResponse(io.BytesIO(audio_bytes), media_type=mime, headers={
        "Content-Length": str(len(audio_bytes)),
        "Cache-Control": "public, max-age=86400",
        "Accept-Ranges": "bytes",
    })
