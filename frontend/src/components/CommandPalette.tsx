'use client'

import { useEffect, useMemo, useRef, useState } from "react"
import { useRouter } from "next/navigation"

import { getEpisodes, runPipeline } from "@/lib/api"
import { Episode } from "@/lib/types"
import { emitToast } from "@/components/Toast"

type Action = {
  id: string
  label: string
  hint?: string
  group: "navigation" | "actions" | "episodes"
  run: () => void
}

export function CommandPalette() {
  const router = useRouter()
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState("")
  const [episodes, setEpisodes] = useState<Episode[]>([])
  const [activeIndex, setActiveIndex] = useState(0)
  const inputRef = useRef<HTMLInputElement | null>(null)
  const listRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const onOpen = () => setOpen(true)
    window.addEventListener("neuropod:palette:open", onOpen)
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement
      const inField = target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)

      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault()
        setOpen((o) => !o)
        return
      }
      if (e.key === "Escape" && open) {
        setOpen(false)
      }
      if (e.key === "/" && !inField && !open) {
        e.preventDefault()
        setOpen(true)
      }
    }
    window.addEventListener("keydown", onKey)
    return () => {
      window.removeEventListener("keydown", onKey)
      window.removeEventListener("neuropod:palette:open", onOpen)
    }
  }, [open])

  useEffect(() => {
    if (!open) {
      setQuery("")
      setActiveIndex(0)
      return
    }
    inputRef.current?.focus()
    if (episodes.length === 0) {
      getEpisodes()
        .then((feed) => setEpisodes(feed.items))
        .catch(() => undefined)
    }
  }, [open])

  const playEpisode = (episode: Episode) => {
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

  const navActions: Action[] = useMemo(() => [
    { id: "nav-home", label: "Go to Feed", hint: "G F", group: "navigation", run: () => router.push("/") },
    { id: "nav-explore", label: "Go to Explore", hint: "G E", group: "navigation", run: () => router.push("/explore") },
    { id: "nav-topics", label: "Go to Topics", hint: "G T", group: "navigation", run: () => router.push("/topics") },
    { id: "nav-subscribe", label: "Go to Feed URL", hint: "G S", group: "navigation", run: () => router.push("/subscribe") },
    {
      id: "act-refresh",
      label: "Refresh feed (run pipeline)",
      hint: "R",
      group: "actions",
      run: async () => {
        try {
          await runPipeline()
          emitToast("Pipeline started", "info")
        } catch (err) {
          emitToast(err instanceof Error ? err.message : "Trigger failed", "error")
        }
      }
    },
    {
      id: "act-shortcuts",
      label: "Show keyboard shortcuts",
      hint: "?",
      group: "actions",
      run: () => window.dispatchEvent(new CustomEvent("neuropod:shortcuts:open"))
    }
  ], [router])

  const episodeActions: Action[] = useMemo(
    () => episodes.map((episode) => ({
      id: `ep-${episode.id}`,
      label: episode.title,
      hint: episode.topic,
      group: "episodes" as const,
      run: () => {
        playEpisode(episode)
        router.push(`/episode/${episode.id}`)
      }
    })),
    [episodes, router]
  )

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    const all = [...navActions, ...episodeActions]
    if (!q) return all
    return all.filter((action) => action.label.toLowerCase().includes(q))
  }, [navActions, episodeActions, query])

  useEffect(() => {
    setActiveIndex(0)
  }, [query])

  useEffect(() => {
    listRef.current?.querySelector<HTMLElement>(`[data-idx="${activeIndex}"]`)?.scrollIntoView({ block: "nearest" })
  }, [activeIndex])

  if (!open) return null

  const handleSelect = (action: Action) => {
    setOpen(false)
    action.run()
  }

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault()
      setActiveIndex((i) => Math.min(filtered.length - 1, i + 1))
    } else if (e.key === "ArrowUp") {
      e.preventDefault()
      setActiveIndex((i) => Math.max(0, i - 1))
    } else if (e.key === "Enter") {
      e.preventDefault()
      const action = filtered[activeIndex]
      if (action) handleSelect(action)
    }
  }

  return (
    <div className="palette-overlay" onClick={() => setOpen(false)}>
      <div className="palette" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Command palette">
        <div className="palette-input-wrap">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="11" cy="11" r="7" />
            <line x1="21" y1="21" x2="16.5" y2="16.5" />
          </svg>
          <input
            ref={inputRef}
            className="palette-input"
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Type a command, episode, or topic..."
            value={query}
          />
          <span className="kbd">esc</span>
        </div>

        <div className="palette-list" ref={listRef}>
          {filtered.length === 0 ? (
            <div className="palette-empty">No matches.</div>
          ) : (
            <PaletteGroups actions={filtered} activeIndex={activeIndex} onSelect={handleSelect} setActiveIndex={setActiveIndex} />
          )}
        </div>

        <div className="palette-footer">
          <span><span className="kbd">↑↓</span> navigate</span>
          <span><span className="kbd">↵</span> select</span>
          <span><span className="kbd">esc</span> close</span>
        </div>
      </div>
    </div>
  )
}

function PaletteGroups({
  actions,
  activeIndex,
  onSelect,
  setActiveIndex
}: {
  actions: Action[]
  activeIndex: number
  onSelect: (a: Action) => void
  setActiveIndex: (i: number) => void
}) {
  const groups: Record<string, { label: string; items: Action[] }> = {
    navigation: { label: "Navigation", items: [] },
    actions: { label: "Actions", items: [] },
    episodes: { label: "Episodes", items: [] }
  }
  actions.forEach((action) => groups[action.group].items.push(action))

  let runningIndex = 0
  return (
    <>
      {Object.entries(groups).map(([key, group]) =>
        group.items.length === 0 ? null : (
          <div className="palette-group" key={key}>
            <div className="palette-group-label">{group.label}</div>
            {group.items.map((action) => {
              const idx = runningIndex++
              return (
                <button
                  key={action.id}
                  className={`palette-item ${idx === activeIndex ? "active" : ""}`}
                  data-idx={idx}
                  onClick={() => onSelect(action)}
                  onMouseEnter={() => setActiveIndex(idx)}
                  type="button"
                >
                  <span className="palette-label">{action.label}</span>
                  {action.hint ? <span className="palette-hint">{action.hint}</span> : null}
                </button>
              )
            })}
          </div>
        )
      )}
    </>
  )
}
