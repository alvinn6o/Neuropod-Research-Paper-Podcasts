import Link from "next/link"

import { DeepDiveChat } from "@/components/DeepDiveChat"
import { PlayButton } from "@/components/PlayButton"
import { getEpisode } from "@/lib/api"
import { splitScript } from "@/lib/script"

type Props = {
  params: Promise<{
    id: string
  }>
}

export default async function EpisodePage({ params }: Props) {
  const { id } = await params
  const episode = await getEpisode(id)
  const minutes = Math.max(1, Math.round(episode.duration_secs / 60))
  const verified = episode.qa_status === "verified"
  const paragraphs = splitScript(episode.script ?? "")

  return (
    <div className="stack-gap">
      <Link className="back-link" href="/">
        ← Back
      </Link>

      <section className="detail-grid">
        <div className="card detail-card">
          <div className="row-between" style={{ marginBottom: 12 }}>
            <div className="tag-list">
              <span className="pill">{episode.topic}</span>
              <span className={verified ? "metric" : "metric metric-warn"}>
                {verified ? "QA OK" : "QA FLAG"}
              </span>
              <span className="pill">{minutes}m</span>
              <span className="pill">{episode.tts_provider}</span>
            </div>
          </div>
          <h1>{episode.title.replace(/: audio brief$/i, "")}</h1>
          <p className="description">{episode.description}</p>

          <div className="meta-row">
            <span>{episode.paper.authors.slice(0, 3).join(", ")}{episode.paper.authors.length > 3 ? " et al." : ""}</span>
            <span>{new Date(episode.paper.published_at).toLocaleDateString()}</span>
            <span>{episode.paper.citation_count} citations</span>
          </div>

          <div className="divider" />

          <div className="row-between">
            <PlayButton episode={episode} />
            <div style={{ display: "flex", gap: 8 }}>
              <a className="button button-secondary" href={episode.paper.pdf_url} rel="noreferrer" target="_blank">
                PDF
              </a>
              <a className="button button-ghost" href={episode.audio_url} rel="noreferrer" target="_blank">
                Audio
              </a>
            </div>
          </div>
        </div>

        <aside className="card column-stack detail-side">
          <span className="label">Abstract</span>
          <p className="abstract-body">{episode.paper.abstract}</p>
          <div className="divider" />
          <span className="label">Categories</span>
          <div className="tag-list">
            {episode.paper.categories.map((cat) => (
              <span className="pill" key={cat}>{cat}</span>
            ))}
          </div>
          <div className="divider" />
          <div className="row-between">
            <span className="label" style={{ margin: 0 }}>Rank score</span>
            <span className="meta-text" style={{ fontSize: 12 }}>{episode.score.toFixed(3)}</span>
          </div>
          <div className="row-between">
            <span className="label" style={{ margin: 0 }}>arXiv ID</span>
            <span className="meta-text" style={{ fontSize: 12 }}>{episode.paper.arxiv_id}</span>
          </div>
        </aside>
      </section>

      <section className="script-card">
        <span className="label">Generated narration</span>
        <div className="script-body">
          {paragraphs.length === 0 ? (
            <p className="meta-text">No script available.</p>
          ) : (
            paragraphs.map((para, idx) => <p key={idx}>{para}</p>)
          )}
        </div>
      </section>

      <DeepDiveChat episodeId={episode.id} />
    </div>
  )
}
