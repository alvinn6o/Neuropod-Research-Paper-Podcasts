'use client'

import { useEffect, useRef, useState } from "react"
import { useRouter } from "next/navigation"

import { getPipelineState, runPipeline } from "@/lib/api"
import { emitToast } from "@/components/Toast"

const WINDOWS = [
  { days: 1, label: "1d" },
  { days: 3, label: "3d" },
  { days: 7, label: "1w" },
  { days: 30, label: "1mo" }
]

const STORAGE_KEY = "neuropod-window-days"

export function RefreshButton() {
  const router = useRouter()
  const [running, setRunning] = useState(false)
  const [windowDays, setWindowDays] = useState<number>(7)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    const saved = window.localStorage.getItem(STORAGE_KEY)
    if (saved) {
      const parsed = Number.parseInt(saved, 10)
      if (Number.isFinite(parsed) && parsed > 0) setWindowDays(parsed)
    }
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [])

  const pickWindow = (days: number) => {
    setWindowDays(days)
    window.localStorage.setItem(STORAGE_KEY, String(days))
  }

  const trigger = async () => {
    if (running) return
    setRunning(true)
    try {
      const result = await runPipeline(windowDays)
      if (!result.queued && result.running) {
        emitToast("Pipeline already running", "info")
      } else {
        emitToast(`Generating from last ${windowDays}d...`, "info")
      }

      pollRef.current = setInterval(async () => {
        try {
          const state = await getPipelineState()
          if (!state.running) {
            if (pollRef.current) clearInterval(pollRef.current)
            pollRef.current = null
            setRunning(false)
            if (state.error) {
              emitToast(`Pipeline failed: ${state.error}`, "error")
            } else {
              emitToast(`Generated ${state.last_count} episode${state.last_count === 1 ? "" : "s"}`, "success")
              router.refresh()
            }
          }
        } catch {
          // ignore
        }
      }, 1500)
    } catch (err) {
      setRunning(false)
      emitToast(err instanceof Error ? err.message : "Pipeline trigger failed", "error")
    }
  }

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
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
          >
            {w.label}
          </button>
        ))}
      </div>
      <button
        className="button button-secondary"
        onClick={trigger}
        disabled={running}
        type="button"
        title={`Run pipeline over last ${windowDays} day${windowDays === 1 ? "" : "s"}`}
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" style={{
          animation: running ? "spin 1s linear infinite" : undefined
        }}>
          <polyline points="23 4 23 10 17 10" />
          <polyline points="1 20 1 14 7 14" />
          <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
        </svg>
        {running ? "Generating..." : "Refresh feed"}
      </button>
    </div>
  )
}
