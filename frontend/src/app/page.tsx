import { EpisodeCard } from "@/components/EpisodeCard"
import { RefreshButton } from "@/components/RefreshButton"
import { getEpisodes, getStatus } from "@/lib/api"
import { relativeTime } from "@/lib/time"

export default async function HomePage() {
  const [feed, status] = await Promise.all([
    getEpisodes(),
    getStatus().catch(() => null)
  ])

  const totalDuration = feed.items.reduce((sum, ep) => sum + ep.duration_secs, 0)
  const verifiedCount = feed.items.filter((ep) => ep.qa_status === "verified").length
  const lastRun = status?.last_pipeline_run ? relativeTime(status.last_pipeline_run) : null

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
          <div className="stat-value">{feed.items.length}</div>
          <span className="stat-label">Episodes</span>
        </div>
        <div className="stat">
          <div className="stat-value">{Math.round(totalDuration / 60)}m</div>
          <span className="stat-label">Total runtime</span>
        </div>
        <div className="stat">
          <div className="stat-value">{verifiedCount}/{feed.items.length}</div>
          <span className="stat-label">QA verified</span>
        </div>
        <div className="stat">
          <div className="stat-value">{feed.topics.length}</div>
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

        {feed.items.length === 0 ? (
          <div className="empty-state">
            <h3>No episodes yet</h3>
            <p>Click Refresh feed to run the pipeline.</p>
          </div>
        ) : (
          <div className="episode-grid" style={{ marginTop: 16 }}>
            {feed.items.map((episode) => (
              <EpisodeCard episode={episode} key={episode.id} />
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
