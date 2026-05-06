'use client'

import { useEffect, useState } from "react"

import { getStatus } from "@/lib/api"
import { StatusResponse } from "@/lib/types"

export function StatusBadge() {
  const [status, setStatus] = useState<StatusResponse | null>(null)

  useEffect(() => {
    getStatus().then(setStatus).catch(() => undefined)
    const interval = setInterval(() => {
      getStatus().then(setStatus).catch(() => undefined)
    }, 30_000)
    return () => clearInterval(interval)
  }, [])

  if (!status) return null

  const providers = status.providers
  const live = providers && providers.llm !== "demo" && providers.tts !== "demo"
  const failures = Object.entries(status.provider_calls || {})
    .filter(([, v]) => v && v.ok === false)
    .map(([k, v]) => `${k}: ${(v.error || "").slice(0, 80)}`)
  const hasFailure = failures.length > 0
  const label = !status.authenticated
    ? "guest"
    : live
      ? "live"
      : "demo"

  const lines = [
    `Auth: ${status.auth_mode}`,
    providers ? `LLM: ${providers.llm}` : "LLM: —",
    providers ? `TTS: ${providers.tts}` : "TTS: —",
    providers ? `Embed: ${providers.embedder}` : "Embed: —",
    `Window: ${status.discovery_window_days}d`,
  ]
  if (failures.length) {
    lines.push("", "Recent failures:", ...failures)
  }

  const dotClass = hasFailure ? "signal-dot warn" : live ? "signal-dot" : "signal-dot warn"

  return (
    <div className="signal-bar" title={lines.join("\n")}>
      <span className={dotClass} />
      {hasFailure ? `${label} ⚠` : label}
    </div>
  )
}
