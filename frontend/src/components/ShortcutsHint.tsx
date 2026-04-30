'use client'

import { useEffect, useState } from "react"

const SHORTCUTS: { key: string; label: string }[] = [
  { key: "⌘K  /  Ctrl+K", label: "Command palette" },
  { key: "/", label: "Focus search" },
  { key: "?", label: "Show shortcuts" },
  { key: "Space", label: "Play / pause" },
  { key: "J  or  ←", label: "Skip back 15s" },
  { key: "L  or  →", label: "Skip forward 30s" },
  { key: "Esc", label: "Close dialogs" }
]

export function ShortcutsHint() {
  const [open, setOpen] = useState(false)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement
      const inField = target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)
      if (e.key === "?" && !inField) {
        setOpen((v) => !v)
      } else if (e.key === "Escape" && open) {
        setOpen(false)
      }
    }
    const onOpen = () => setOpen(true)
    window.addEventListener("keydown", onKey)
    window.addEventListener("neuropod:shortcuts:open", onOpen)
    return () => {
      window.removeEventListener("keydown", onKey)
      window.removeEventListener("neuropod:shortcuts:open", onOpen)
    }
  }, [open])

  if (!open) return null

  return (
    <div className="palette-overlay" onClick={() => setOpen(false)}>
      <div className="palette" style={{ maxWidth: 420 }} onClick={(e) => e.stopPropagation()} role="dialog">
        <div className="palette-input-wrap" style={{ borderBottom: "1px solid var(--border)" }}>
          <strong style={{ flex: 1, fontSize: 13, fontWeight: 500 }}>Keyboard shortcuts</strong>
          <span className="kbd">esc</span>
        </div>
        <div className="palette-list">
          {SHORTCUTS.map((s) => (
            <div className="palette-item" key={s.key}>
              <span className="palette-label">{s.label}</span>
              <span className="palette-hint">{s.key}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
