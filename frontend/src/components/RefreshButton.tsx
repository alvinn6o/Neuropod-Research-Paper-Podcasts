'use client'

import { useEffect, useRef, useState } from "react"
import { useRouter } from "next/navigation"

import { getPipelineJob, runPipeline } from "@/lib/api"
import { emitToast } from "@/components/Toast"

const WINDOWS = [
  { days: 1, label: "1d" },
  { days: 3, label: "3d" },
  { days: 7, label: "1w" },
  { days: 30, label: "1mo" },
]

// Per-run cap — matches backend MAX_EPISODES_PER_RUN. Backend re-validates,
// so changing this without the backend just shows extra options that get
// rejected at the API layer (treat backend as source of truth).
const EPISODE_COUNTS = [1, 2, 3] as const
const DEFAULT_EPISODES = 3

const WINDOW_STORAGE_KEY = "neuropod-window-days"
const EPISODES_STORAGE_KEY = "neuropod-episodes-per-run"

export function RefreshButton() {
  const router = useRouter()
  const [running, setRunning] = useState(false)
  const [windowDays, setWindowDays] = useState<number>(7)
  const [episodesPerRun, setEpisodesPerRun] = useState<number>(DEFAULT_EPISODES)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    const savedWindow = window.localStorage.getItem(WINDOW_STORAGE_KEY)
    if (savedWindow) {
      const parsed = Number.parseInt(savedWindow, 10)
      if (Number.isFinite(parsed) && parsed > 0) setWindowDays(parsed)
    }
    const savedEpisodes = window.localStorage.getItem(EPISODES_STORAGE_KEY)
    if (savedEpisodes) {
      const parsed = Number.parseInt(savedEpisodes, 10)
      if (EPISODE_COUNTS.includes(parsed as typeof EPISODE_COUNTS[number])) {
        setEpisodesPerRun(parsed)
      }
    }
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [])

  const pickWindow = (days: number) => {
    setWindowDays(days)
    window.localStorage.setItem(WINDOW_STORAGE_KEY, String(days))
  }

  const pickEpisodes = (count: number) => {
    setEpisodesPerRun(count)
    window.localStorage.setItem(EPISODES_STORAGE_KEY, String(count))
  }

  const trigger = async () => {
    if (running) return
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
    setRunning(true)
    try {
      const job = await runPipeline({ windowDays, episodes: episodesPerRun })
      const epLabel = episodesPerRun === 1 ? "episode" : "episodes"
      emitToast(`Generating up to ${episodesPerRun} ${epLabel} from last ${windowDays}d…`, "info")

      const maxPolls = Math.floor((10 * 60 * 1000) / 1500)
      let polls = 0
      let consecutiveFailures = 0

      pollRef.current = setInterval(async () => {
        polls += 1
        if (polls > maxPolls) {
          if (pollRef.current) clearInterval(pollRef.current)
          pollRef.current = null
          setRunning(false)
          emitToast("Pipeline taking longer than expected. Check logs.", "error")
          return
        }
        try {
          const next = await getPipelineJob(job.id)
          consecutiveFailures = 0
          if (next.status === "done" || next.status === "error") {
            if (pollRef.current) clearInterval(pollRef.current)
            pollRef.current = null
            setRunning(false)
            if (next.status === "error") {
              emitToast(`Pipeline failed: ${next.error || "unknown error"}`, "error")
            } else {
              const generated = next.result_count
              const skipped = next.skipped_count ?? 0
              const newMsg = `Generated ${generated} new episode${generated === 1 ? "" : "s"}`
              const skippedMsg = skipped > 0 ? ` (${skipped} already in feed)` : ""
              emitToast(`${newMsg}${skippedMsg}`, "success")
              if (generated > 0) {
                router.refresh()
                window.location.reload()
              }
            }
          }
        } catch {
          consecutiveFailures += 1
          if (consecutiveFailures >= 5) {
            if (pollRef.current) clearInterval(pollRef.current)
            pollRef.current = null
            setRunning(false)
            emitToast("Lost connection to API while polling. Try again.", "error")
          }
        }
      }, 1500)
    } catch (err) {
      setRunning(false)
      emitToast(err instanceof Error ? err.message : "Pipeline trigger failed", "error")
    }
  }

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
      <div className="window-selector" role="radiogroup" aria-label="Discovery window">
        {WINDOWS.map((w) => (
          <button
            key={w.days}
            type="button"
            role="radio"
            aria-checked={windowDays === w.days}
            className={windowDays === w.days ? "active" : ""}
            onClick={() => pickWindow(w.days)}
            disabled={running}
            title={`Search papers from the last ${w.days} day${w.days === 1 ? "" : "s"}`}
          >
            {w.label}
          </button>
        ))}
      </div>
      <div className="window-selector" role="radiogroup" aria-label="Episodes per run">
        {EPISODE_COUNTS.map((n) => (
          <button
            key={n}
            type="button"
            role="radio"
            aria-checked={episodesPerRun === n}
            className={episodesPerRun === n ? "active" : ""}
            onClick={() => pickEpisodes(n)}
            disabled={running}
            title={`Generate up to ${n} episode${n === 1 ? "" : "s"} per run`}
          >
            {n}ep
          </button>
        ))}
      </div>
      <button
        className="button button-secondary"
        onClick={trigger}
        disabled={running}
        type="button"
        title={`Run pipeline over last ${windowDays} day${windowDays === 1 ? "" : "s"} for up to ${episodesPerRun} episode${episodesPerRun === 1 ? "" : "s"}`}
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" style={{
          animation: running ? "spin 1s linear infinite" : undefined
        }}>
          <polyline points="23 4 23 10 17 10" />
          <polyline points="1 20 1 14 7 14" />
          <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
        </svg>
        {running ? "Generating…" : "Refresh feed"}
      </button>
    </div>
  )
}
