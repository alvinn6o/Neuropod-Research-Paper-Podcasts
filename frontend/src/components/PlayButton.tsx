'use client'

import { Episode } from "@/lib/types"

type Props = {
  episode: Episode
  variant?: "primary" | "secondary"
}

export function PlayButton({ episode, variant = "primary" }: Props) {
  const hasAudio = episode.audio_ready && Boolean(episode.audio_url)

  const play = () => {
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
    <button
      className={`button button-${variant}`}
      disabled={!hasAudio}
      onClick={play}
      title={hasAudio ? "Play episode" : "Generate audio first"}
      type="button"
    >
      <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
        <path d="M7 5v14l12-7z" />
      </svg>
      {hasAudio ? "Play episode" : "Audio not generated"}
    </button>
  )
}
