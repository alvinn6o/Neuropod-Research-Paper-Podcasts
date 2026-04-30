from __future__ import annotations

from datetime import datetime
from email.utils import format_datetime
from xml.sax.saxutils import escape

from fastapi import APIRouter, Depends, Request, Response

from ..dependencies import get_store
from ..storage import DemoStore

router = APIRouter(prefix="/feed", tags=["feed"])


def _build_manual_feed(request: Request, episodes: list[dict], user_slug: str) -> str:
    base = request.base_url._url.rstrip("/")
    items = []
    for episode in episodes:
        pub_date = format_datetime(datetime.fromisoformat(episode["created_at"].replace("Z", "+00:00")))
        audio_url = f"{base}/episodes/{episode['id']}/audio"
        page_url = f"{base}/episodes/{episode['id']}"
        description = escape(episode["description"])
        title = escape(episode["title"])
        items.append(
            f"""
            <item>
              <title>{title}</title>
              <link>{page_url}</link>
              <guid>{episode['id']}</guid>
              <description>{description}</description>
              <pubDate>{pub_date}</pubDate>
              <enclosure url="{audio_url}" type="audio/mpeg" length="0" />
            </item>
            """.strip()
        )

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Neuropod — {escape(user_slug)}</title>
    <link>{base}</link>
    <description>Daily research papers, distilled into audio.</description>
    {''.join(items)}
  </channel>
</rss>
"""


@router.get("/{user_slug}")
def get_feed(
    user_slug: str,
    request: Request,
    store: DemoStore = Depends(get_store),
) -> Response:
    episodes = store.list_episodes(limit=25)

    try:
        from feedgen.feed import FeedGenerator

        fg = FeedGenerator()
        fg.title(f"Neuropod — {user_slug}")
        fg.link(href=str(request.base_url).rstrip("/"), rel="alternate")
        fg.description("Daily research papers, distilled into audio.")
        fg.language("en")

        for episode in episodes:
            entry = fg.add_entry()
            entry.id(episode["id"])
            entry.title(episode["title"])
            entry.description(episode["description"])
            entry.pubDate(datetime.fromisoformat(episode["created_at"].replace("Z", "+00:00")))
            entry.enclosure(
                str(request.base_url).rstrip("/") + f"/episodes/{episode['id']}/audio",
                0,
                "audio/mpeg",
            )
            entry.link(href=str(request.base_url).rstrip("/") + f"/episodes/{episode['id']}")

        return Response(content=fg.rss_str(pretty=True), media_type="application/rss+xml")
    except Exception:
        return Response(
            content=_build_manual_feed(request, episodes, user_slug),
            media_type="application/rss+xml",
        )
