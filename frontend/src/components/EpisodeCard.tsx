'use client'

import { useRouter } from "next/navigation"

import { Episode } from "@/lib/types"
import { formatDuration, relativeTime } from "@/lib/time"

type Props = {
  episode: Episode
}

export function EpisodeCard({ episode }: Props) {
  const router = useRouter()
  const verified = episode.qa_status === "verified"
  const created = episode.created_at ? relativeTime(episode.created_at) : ""
  const hasAudio = episode.audio_ready && Boolean(episode.audio_url)

  const play = (e: React.MouseEvent) => {
    e.stopPropagation()
    if (!hasAudio || !episode.audio_url) return
    window.dispatchEvent(
      new CustomEvent("neuropod:play", {
        detail: {
          id: episode.id,
          title: episode.title,
          description: episode.description,
          audioUrl: episode.audio_url,
          topic: episode.topic,
          duration: episode.duration_secs
        }
      })
    )
  }

  return (
    <article
      className="card card-interactive episode-card"
      onClick={() => router.push(`/episode/${episode.id}`)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter") router.push(`/episode/${episode.id}`)
      }}
    >
      <div className="episode-card-header">
        <span className="pill">{episode.topic}</span>
        <span className={verified ? "metric" : "metric metric-warn"}>
          {verified ? "QA OK" : "QA FLAG"}
        </span>
      </div>
      <h3>{episode.title.replace(/: audio brief$/i, "")}</h3>
      <p>{episode.description}</p>
      <div className="episode-card-footer">
        <button
          className="icon-button"
          disabled={!hasAudio}
          onClick={play}
          title={hasAudio ? "Play episode" : "Open episode to generate audio"}
          type="button"
          aria-label={hasAudio ? "Play episode" : "Audio not generated"}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
            <path d="M7 5v14l12-7z" />
          </svg>
        </button>
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          {created ? <span className="meta-text">{created}</span> : null}
          <span className="meta-text">{formatDuration(episode.duration_secs)}</span>
          {!hasAudio ? <span className="meta-text">script</span> : null}
          <span className="meta-text">{episode.paper.citation_count} cites</span>
        </div>
      </div>
    </article>
  )
}
