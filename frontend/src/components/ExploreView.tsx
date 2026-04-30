'use client'

import { useMemo, useState } from "react"

import { EpisodeCard } from "@/components/EpisodeCard"
import { Episode } from "@/lib/types"

type Props = {
  episodes: Episode[]
  topics: string[]
}

type SortKey = "recent" | "citations" | "score" | "duration"

const SORT_LABELS: Record<SortKey, string> = {
  recent: "Recent",
  citations: "Citations",
  score: "Rank",
  duration: "Duration"
}

export function ExploreView({ episodes, topics }: Props) {
  const [query, setQuery] = useState("")
  const [activeTopic, setActiveTopic] = useState<string | null>(null)
  const [sort, setSort] = useState<SortKey>("citations")

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    let list = episodes.filter((episode) => {
      if (activeTopic && !episode.topic.toLowerCase().includes(activeTopic.toLowerCase()) &&
          !episode.paper.categories.some((c) => c.toLowerCase().includes(activeTopic.toLowerCase()))) {
        return false
      }
      if (!q) return true
      return (
        episode.title.toLowerCase().includes(q) ||
        episode.description.toLowerCase().includes(q) ||
        episode.paper.abstract.toLowerCase().includes(q) ||
        episode.paper.authors.some((a) => a.toLowerCase().includes(q))
      )
    })

    list = [...list].sort((a, b) => {
      switch (sort) {
        case "citations":
          return b.paper.citation_count - a.paper.citation_count
        case "score":
          return b.score - a.score
        case "duration":
          return b.duration_secs - a.duration_secs
        case "recent":
        default:
          return new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      }
    })

    return list
  }, [episodes, query, activeTopic, sort])

  return (
    <div className="stack-gap">
      <section className="hero">
        <h1>Explore</h1>
        <p>Search across episodes, papers, and authors.</p>
      </section>

      <div className="search-bar">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="11" cy="11" r="7" />
          <line x1="21" y1="21" x2="16.5" y2="16.5" />
        </svg>
        <input
          autoFocus
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search title, abstract, or author..."
          value={query}
        />
        {query ? (
          <button className="icon-button" onClick={() => setQuery("")} type="button" aria-label="Clear">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        ) : null}
      </div>

      <div className="row-between" style={{ flexWrap: "wrap", gap: 12 }}>
        <div className="filter-row">
          <button
            className={`chip ${activeTopic === null ? "chip-active" : ""}`}
            onClick={() => setActiveTopic(null)}
            type="button"
          >
            All
          </button>
          {topics.map((topic) => (
            <button
              className={`chip ${activeTopic === topic ? "chip-active" : ""}`}
              key={topic}
              onClick={() => setActiveTopic(activeTopic === topic ? null : topic)}
              type="button"
            >
              {topic}
            </button>
          ))}
        </div>

        <div className="filter-row">
          <span className="meta-text">Sort</span>
          {(Object.keys(SORT_LABELS) as SortKey[]).map((key) => (
            <button
              className={`chip ${sort === key ? "chip-active" : ""}`}
              key={key}
              onClick={() => setSort(key)}
              type="button"
            >
              {SORT_LABELS[key]}
            </button>
          ))}
        </div>
      </div>

      <div className="section-heading">
        <h2>{filtered.length} {filtered.length === 1 ? "result" : "results"}</h2>
      </div>

      {filtered.length === 0 ? (
        <div className="empty-state">
          <h3>No matches</h3>
          <p>Try a different query or clear filters.</p>
        </div>
      ) : (
        <div className="episode-grid">
          {filtered.map((episode) => (
            <EpisodeCard episode={episode} key={episode.id} />
          ))}
        </div>
      )}
    </div>
  )
}
