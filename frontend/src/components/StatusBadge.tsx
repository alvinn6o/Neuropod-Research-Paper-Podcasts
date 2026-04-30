import { getStatus } from "@/lib/api"

export async function StatusBadge() {
  let live = false
  let label = "demo"
  let title = ""
  let hasFailure = false

  try {
    const status = await getStatus()
    live = status.providers.llm !== "demo" && status.providers.tts !== "demo"
    label = live ? "live" : "demo"

    const failures = Object.entries(status.provider_calls || {})
      .filter(([, v]) => v && v.ok === false)
      .map(([k, v]) => `${k}: ${(v.error || "").slice(0, 80)}`)

    hasFailure = failures.length > 0

    const lines = [
      `LLM: ${status.providers.llm}`,
      `TTS: ${status.providers.tts}`,
      `Embed: ${status.providers.embedder}`,
      `Window: ${status.discovery_window_days}d`,
    ]
    if (failures.length) {
      lines.push("")
      lines.push("Recent failures:")
      lines.push(...failures)
    }
    title = lines.join("\n")
  } catch {
    label = "offline"
    title = "API unreachable"
  }

  const dotClass = hasFailure ? "signal-dot warn" : live ? "signal-dot" : "signal-dot warn"

  return (
    <div className="signal-bar" title={title}>
      <span className={dotClass} />
      {hasFailure ? `${label} ⚠` : label}
    </div>
  )
}
