from __future__ import annotations

import uuid
from datetime import datetime, timezone
from email.utils import format_datetime
from xml.sax.saxutils import escape

from fastapi import APIRouter, HTTPException, Request, Response

from .. import audio_store, store_db
from ..db import get_user_by_slug

router = APIRouter(prefix="/feed", tags=["feed"])


def _episode_to_item(request: Request, episode: dict) -> str:
    base = str(request.base_url).rstrip("/")
    audio_url = audio_store.url_for(episode.get("audio_key") or "") or \
        f"{base}/episodes/{episode['id']}/audio?key={episode.get('audio_key', '')}"
    page_url = f"{base}/episodes/{episode['id']}"

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
          <enclosure url="{escape(audio_url)}" type="{escape(episode.get('audio_mime', 'audio/mpeg'))}" length="0" />
        </item>""".strip()


@router.get("/{user_slug}")
def get_feed(user_slug: str, request: Request) -> Response:
    user = get_user_by_slug(user_slug)
    if not user:
        raise HTTPException(status_code=404, detail="feed not found")

    episodes = store_db.list_episodes(uuid.UUID(str(user["id"])), limit=50)
    base = str(request.base_url).rstrip("/")
    title = f"Neuropod — {escape(user.get('display_name') or user_slug)}"
    items = "\n".join(_episode_to_item(request, ep) for ep in episodes)

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
