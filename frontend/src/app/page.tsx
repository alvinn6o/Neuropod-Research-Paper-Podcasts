'use client'

import { useEffect, useState } from "react"

import { AuthGuard } from "@/components/AuthGuard"
import { EpisodeCard } from "@/components/EpisodeCard"
import { RefreshButton } from "@/components/RefreshButton"
import { getEpisodes, getStatus } from "@/lib/api"
import { Episode, StatusResponse } from "@/lib/types"
import { relativeTime } from "@/lib/time"

export default function HomePage() {
  return (
    <AuthGuard requireKeys>
      {() => <FeedView />}
    </AuthGuard>
  )
}

function FeedView() {
  const [items, setItems] = useState<Episode[]>([])
  const [topics, setTopics] = useState<string[]>([])
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [loading, setLoading] = useState(true)

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

  const totalDuration = items.reduce((sum, ep) => sum + ep.duration_secs, 0)
  const verifiedCount = items.filter((ep) => ep.qa_status === "verified").length
  const lastRun = status?.last_job?.finished_at ? relativeTime(status.last_job.finished_at) : null

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

      <section>
        <div className="section-heading">
          <h2>Latest</h2>
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
            <p>Click Refresh feed to run the pipeline with your topics.</p>
          </div>
        ) : (
          <div className="episode-grid" style={{ marginTop: 16 }}>
            {items.map((episode) => (
              <EpisodeCard episode={episode} key={episode.id} />
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
