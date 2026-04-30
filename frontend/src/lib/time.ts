export function relativeTime(input: string | Date): string {
  const date = typeof input === "string" ? new Date(input) : input
  const seconds = Math.round((Date.now() - date.getTime()) / 1000)

  if (!Number.isFinite(seconds)) return ""
  if (Math.abs(seconds) < 5) return "just now"

  const abs = Math.abs(seconds)
  const suffix = seconds >= 0 ? " ago" : " from now"
  let value: number
  let unit: string

  if (abs < 60) { value = abs; unit = "s" }
  else if (abs < 3600) { value = Math.round(abs / 60); unit = "m" }
  else if (abs < 86400) { value = Math.round(abs / 3600); unit = "h" }
  else if (abs < 604800) { value = Math.round(abs / 86400); unit = "d" }
  else if (abs < 2629800) { value = Math.round(abs / 604800); unit = "w" }
  else if (abs < 31557600) { value = Math.round(abs / 2629800); unit = "mo" }
  else { value = Math.round(abs / 31557600); unit = "y" }

  return `${value}${unit}${suffix}`
}

export function formatDuration(secs: number): string {
  if (!Number.isFinite(secs) || secs <= 0) return "0m"
  const minutes = Math.round(secs / 60)
  if (minutes < 60) return `${minutes}m`
  const h = Math.floor(minutes / 60)
  const m = minutes % 60
  return m === 0 ? `${h}h` : `${h}h${m}m`
}
