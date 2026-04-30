'use client'

export function CommandTrigger() {
  const open = () => window.dispatchEvent(new CustomEvent("neuropod:palette:open"))
  return (
    <button className="command-trigger" onClick={open} type="button" aria-label="Open command palette">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="11" cy="11" r="7" />
        <line x1="21" y1="21" x2="16.5" y2="16.5" />
      </svg>
      <span>Search</span>
      <span className="kbd">⌘K</span>
    </button>
  )
}
