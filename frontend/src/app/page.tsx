'use client'

import { useEffect, useMemo, useState } from "react"

import { AuthGuard } from "@/components/AuthGuard"
import { EpisodeCard } from "@/components/EpisodeCard"
import { RefreshButton } from "@/components/RefreshButton"
import { getEpisodes, getStatus } from "@/lib/api"
import { Episode, StatusResponse } from "@/lib/types"
import { relativeTime } from "@/lib/time"

type SortKey = "recent" | "citations" | "score" | "duration"
const SORT_LABELS: Record<SortKey, string> = {
  recent: "Recent",
  citations: "Citations",
  score: "Rank",
  duration: "Duration",
}

export default function HomePage() {
  return (
    <AuthGuard>
      {() => <FeedView />}
    </AuthGuard>
  )
}

function FeedView() {
  const [items, setItems] = useState<Episode[]>([])
  const [topics, setTopics] = useState<string[]>([])
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [loading, setLoading] = useState(true)

  const [query, setQuery] = useState("")
  const [activeTopic, setActiveTopic] = useState<string | null>(null)
  const [sort, setSort] = useState<SortKey>("recent")

  useEffect(() => {
    let cancelled = false
    Promise.all([getEpisodes(), getStatus().catch(() => null)])
      .then(([feed, st]) => {
        if (cancelled) return
        setItems(feed.items)
        setTopics(feed.topics)
        setStatus(st as StatusResponse | null)
      })
      .finally(() => !cancelled && setLoading(false))
    return () => { cancelled = true }
  }, [])

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    let list = items.filter((episode) => {
      if (
        activeTopic &&
        !episode.topic.toLowerCase().includes(activeTopic.toLowerCase()) &&
        !episode.paper.categories.some((c) => c.toLowerCase().includes(activeTopic.toLowerCase()))
      ) return false
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
        case "citations": return b.paper.citation_count - a.paper.citation_count
        case "score": return b.score - a.score
        case "duration": return b.duration_secs - a.duration_secs
        case "recent":
        default:
          return new Date(b.created_at ?? 0).getTime() - new Date(a.created_at ?? 0).getTime()
      }
    })

    return list
  }, [items, query, activeTopic, sort])

  const totalDuration = items.reduce((sum, ep) => sum + ep.duration_secs, 0)
  const verifiedCount = items.filter((ep) => ep.qa_status === "verified").length
  const lastRun = status?.last_job?.finished_at ? relativeTime(status.last_job.finished_at) : null
  const hasActiveFilter = query.length > 0 || activeTopic !== null

  return (
    <div className="stack-gap">
      <section className="hero">
        <div className="row-between" style={{ alignItems: "flex-end", flexWrap: "wrap", gap: 16 }}>
          <div>
            <h1>Today&apos;s feed</h1>
            <p>Recent papers, ranked and narrated. Source attached, follow-ups grounded.</p>
          </div>
          <RefreshButton />
        </div>
      </section>

      <div className="stat-row">
        <div className="stat">
          <div className="stat-value">{items.length}</div>
          <span className="stat-label">Episodes</span>
        </div>
        <div className="stat">
          <div className="stat-value">{Math.round(totalDuration / 60)}m</div>
          <span className="stat-label">Total runtime</span>
        </div>
        <div className="stat">
          <div className="stat-value">{verifiedCount}/{items.length || "·"}</div>
          <span className="stat-label">QA verified</span>
        </div>
        <div className="stat">
          <div className="stat-value">{topics.length}</div>
          <span className="stat-label">Topics tracked</span>
        </div>
      </div>

      {items.length > 0 ? (
        <>
          <div className="search-bar">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="7" />
              <line x1="21" y1="21" x2="16.5" y2="16.5" />
            </svg>
            <input
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search title, abstract, or author…"
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

          {topics.length > 0 || true ? (
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
          ) : null}
        </>
      ) : null}

      <section>
        <div className="section-heading">
          <h2>
            {hasActiveFilter
              ? `${filtered.length} ${filtered.length === 1 ? "result" : "results"}`
              : "Latest"}
          </h2>
          <div className="section-meta">
            {lastRun ? <span>generated {lastRun}</span> : null}
          </div>
        </div>

        {loading ? (
          <div className="episode-grid" style={{ marginTop: 16 }}>
            {[0, 1, 2].map((i) => (
              <div key={i} className="card" style={{ height: 160 }}>
                <div className="skeleton" style={{ height: 14, width: "40%" }} />
                <div className="skeleton" style={{ height: 18, width: "85%", marginTop: 12 }} />
                <div className="skeleton" style={{ height: 14, width: "92%", marginTop: 8 }} />
              </div>
            ))}
          </div>
        ) : items.length === 0 ? (
          <div className="empty-state">
            <h3>No episodes yet</h3>
            <p>Add a topic in <a href="/topics">Topics</a>, then click Refresh feed.</p>
          </div>
        ) : filtered.length === 0 ? (
          <div className="empty-state">
            <h3>No matches</h3>
            <p>Try a different query or clear the topic filter.</p>
          </div>
        ) : (
          <div className="episode-grid" style={{ marginTop: 16 }}>
            {filtered.map((episode) => (
              <EpisodeCard episode={episode} key={episode.id} />
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
