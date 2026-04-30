'use client'

import { useCallback, useEffect, useRef, useState } from "react"

import { apiBaseUrl, postFeedback } from "@/lib/api"
import { emitToast } from "@/components/Toast"

type PlayerEpisode = {
  id: string
  title: string
  description: string
  audioUrl: string
  topic?: string
  duration?: number
}

declare global {
  interface WindowEventMap {
    "neuropod:play": CustomEvent<PlayerEpisode>
  }
}

const STORAGE_KEY = "neuropod-active-episode"
const SPEEDS = [0.85, 1, 1.25, 1.5, 1.75, 2] as const

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00"
  const total = Math.floor(seconds)
  const m = Math.floor(total / 60)
  const s = total % 60
  return `${m}:${String(s).padStart(2, "0")}`
}

export function AudioPlayer() {
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const scrubberRef = useRef<HTMLDivElement | null>(null)
  const [episode, setEpisode] = useState<PlayerEpisode | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [speed, setSpeed] = useState(1)
  const [dragging, setDragging] = useState(false)
  const [buffering, setBuffering] = useState(false)
  const [errored, setErrored] = useState(false)
  const lastEpisodeIdRef = useRef<string | null>(null)

  useEffect(() => {
    const saved = window.localStorage.getItem(STORAGE_KEY)
    if (saved) {
      try {
        setEpisode(JSON.parse(saved) as PlayerEpisode)
      } catch {
        window.localStorage.removeItem(STORAGE_KEY)
      }
    }

    const handlePlay = (event: WindowEventMap["neuropod:play"]) => {
      setEpisode(event.detail)
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(event.detail))
    }

    window.addEventListener("neuropod:play", handlePlay as EventListener)
    return () => window.removeEventListener("neuropod:play", handlePlay as EventListener)
  }, [])

  useEffect(() => {
    if (!episode || !audioRef.current) return
    const audio = audioRef.current
    const isNew = lastEpisodeIdRef.current !== episode.id
    lastEpisodeIdRef.current = episode.id

    if (isNew) {
      setErrored(false)
      setBuffering(true)
      setCurrentTime(0)
      audio.src = episode.audioUrl.startsWith("http")
        ? episode.audioUrl
        : `${apiBaseUrl()}${episode.audioUrl}`
      audio.playbackRate = speed
      audio
        .play()
        .then(() => {
          setIsPlaying(true)
          postFeedback(episode.id, "play", 0).catch(() => undefined)
        })
        .catch(() => {
          setIsPlaying(false)
        })
    }
  }, [episode])

  useEffect(() => {
    if (audioRef.current) audioRef.current.playbackRate = speed
  }, [speed])

  const togglePlay = useCallback(() => {
    if (!episode || !audioRef.current || errored) return
    const audio = audioRef.current
    if (audio.paused) {
      audio.play().catch(() => undefined)
    } else {
      audio.pause()
    }
  }, [episode, errored])

  const skip = useCallback((seconds: number) => {
    if (!audioRef.current) return
    audioRef.current.currentTime = Math.max(
      0,
      Math.min(audioRef.current.duration || 0, audioRef.current.currentTime + seconds)
    )
  }, [])

  const seekFromX = useCallback((clientX: number) => {
    if (!scrubberRef.current || !audioRef.current || !duration) return
    const rect = scrubberRef.current.getBoundingClientRect()
    const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
    audioRef.current.currentTime = ratio * duration
    setCurrentTime(ratio * duration)
  }, [duration])

  useEffect(() => {
    if (!dragging) return
    const onMove = (e: MouseEvent) => seekFromX(e.clientX)
    const onTouchMove = (e: TouchEvent) => {
      if (e.touches.length > 0) seekFromX(e.touches[0].clientX)
    }
    const stop = () => setDragging(false)
    window.addEventListener("mousemove", onMove)
    window.addEventListener("touchmove", onTouchMove, { passive: true })
    window.addEventListener("mouseup", stop)
    window.addEventListener("touchend", stop)
    window.addEventListener("touchcancel", stop)
    return () => {
      window.removeEventListener("mousemove", onMove)
      window.removeEventListener("touchmove", onTouchMove)
      window.removeEventListener("mouseup", stop)
      window.removeEventListener("touchend", stop)
      window.removeEventListener("touchcancel", stop)
    }
  }, [dragging, seekFromX])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) {
        return
      }
      if (e.code === "Space") {
        if (!episode) return
        e.preventDefault()
        togglePlay()
      } else if (e.key === "j" || e.key === "ArrowLeft") {
        if (!episode) return
        skip(-15)
      } else if (e.key === "l" || e.key === "ArrowRight") {
        if (!episode) return
        skip(30)
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [togglePlay, skip, episode])

  const cycleSpeed = () => {
    const idx = SPEEDS.indexOf(speed as typeof SPEEDS[number])
    setSpeed(SPEEDS[(idx + 1) % SPEEDS.length])
  }

  if (!episode) {
    return null
  }

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0
  const playLabel = errored ? "Audio failed" : isPlaying ? "Pause" : "Play"

  return (
    <div className="player-shell">
      <div className="player-inner">
        <div className="player-meta">
          <strong>{episode.title}</strong>
          <span>
            {errored ? "audio unavailable" : buffering && !isPlaying ? "loading…" : episode.topic ?? "now playing"}
          </span>
        </div>

        <div className="player-controls">
          <button className="icon-button" onClick={() => skip(-15)} aria-label="Back 15 seconds" type="button" disabled={errored}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="11 17 6 12 11 7" />
              <polyline points="18 17 13 12 18 7" />
            </svg>
          </button>
          <button
            className={`player-play ${errored ? "errored" : ""}`}
            onClick={togglePlay}
            aria-label={playLabel}
            type="button"
            disabled={errored}
          >
            {errored ? (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            ) : buffering && !isPlaying ? (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" style={{ animation: "spin 0.9s linear infinite" }}>
                <path d="M21 12a9 9 0 1 1-6.22-8.56" />
              </svg>
            ) : isPlaying ? (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                <rect x="6" y="5" width="4" height="14" rx="1" />
                <rect x="14" y="5" width="4" height="14" rx="1" />
              </svg>
            ) : (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                <path d="M7 5v14l12-7z" />
              </svg>
            )}
          </button>
          <button className="icon-button" onClick={() => skip(30)} aria-label="Forward 30 seconds" type="button" disabled={errored}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="13 17 18 12 13 7" />
              <polyline points="6 17 11 12 6 7" />
            </svg>
          </button>
          <button className="speed-button" onClick={cycleSpeed} type="button" aria-label="Playback speed">
            {speed}×
          </button>
        </div>

        <div className="player-progress">
          <span className="player-time">{formatTime(currentTime)}</span>
          <div
            className={`scrubber ${dragging ? "dragging" : ""}`}
            ref={scrubberRef}
            onMouseDown={(e) => {
              setDragging(true)
              seekFromX(e.clientX)
            }}
            onTouchStart={(e) => {
              setDragging(true)
              if (e.touches.length > 0) seekFromX(e.touches[0].clientX)
            }}
            role="slider"
            aria-valuemin={0}
            aria-valuemax={duration}
            aria-valuenow={currentTime}
            aria-label="Seek"
          >
            <div className="scrubber-fill" style={{ width: `${progress}%` }} />
            <div className="scrubber-thumb" style={{ left: `${progress}%` }} />
          </div>
          <span className="player-time">{formatTime(duration)}</span>
        </div>
      </div>

      <audio
        ref={audioRef}
        onPlay={() => {
          setIsPlaying(true)
          setBuffering(false)
        }}
        onPause={(event) => {
          setIsPlaying(false)
          if (!event.currentTarget.ended) {
            postFeedback(episode.id, "pause", event.currentTarget.currentTime).catch(() => undefined)
          }
        }}
        onWaiting={() => setBuffering(true)}
        onPlaying={() => setBuffering(false)}
        onCanPlay={() => setBuffering(false)}
        onTimeUpdate={(event) => !dragging && setCurrentTime(event.currentTarget.currentTime)}
        onLoadedMetadata={(event) => setDuration(event.currentTarget.duration || 0)}
        onError={() => {
          setErrored(true)
          setIsPlaying(false)
          setBuffering(false)
          emitToast("Audio failed to load — check provider keys or try Refresh feed", "error")
        }}
        onEnded={(event) => {
          setIsPlaying(false)
          postFeedback(episode.id, "complete", event.currentTarget.duration).catch(() => undefined)
        }}
      />
    </div>
  )
}
