'use client'

import Link from "next/link"
import { use, useEffect, useState } from "react"

import { AuthGuard } from "@/components/AuthGuard"
import { DeepDiveChat } from "@/components/DeepDiveChat"
import { PlayButton } from "@/components/PlayButton"
import { emitToast } from "@/components/Toast"
import { generateEpisodeAudio, getEpisode } from "@/lib/api"
import { splitScript } from "@/lib/script"
import { Episode } from "@/lib/types"

type Props = { params: Promise<{ id: string }> }

export default function EpisodePage({ params }: Props) {
  const { id } = use(params)
  return (
    <AuthGuard>
      {() => <EpisodeView id={id} />}
    </AuthGuard>
  )
}

function EpisodeView({ id }: { id: string }) {
  const [episode, setEpisode] = useState<Episode | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [audioBusy, setAudioBusy] = useState(false)

  useEffect(() => {
    getEpisode(id).then(setEpisode).catch((err) => setError(err.message))
  }, [id])

  if (error) return <div className="empty-state"><h3>Couldn&apos;t load episode</h3><p>{error}</p></div>
  if (!episode) return <div className="card"><div className="skeleton" style={{ height: 200 }} /></div>

  const minutes = Math.max(1, Math.round(episode.duration_secs / 60))
  const verified = episode.qa_status === "verified"
  const hasAudio = episode.audio_ready && Boolean(episode.audio_url)
  const paragraphs = splitScript(episode.script ?? "")

  const generateAudio = async () => {
    if (audioBusy) return
    setAudioBusy(true)
    try {
      const updated = await generateEpisodeAudio(episode.id)
      setEpisode(updated)
      emitToast("Audio generated", "success")
    } catch (err) {
      emitToast(err instanceof Error ? err.message : "Audio generation failed", "error")
    } finally {
      setAudioBusy(false)
    }
  }

  return (
    <div className="stack-gap">
      <Link className="back-link" href="/">Back</Link>

      <section className="detail-grid">
        <div className="card detail-card">
          <div className="row-between" style={{ marginBottom: 12 }}>
            <div className="tag-list">
              <span className="pill">{episode.topic}</span>
              <span className={verified ? "metric" : "metric metric-warn"}>
                {verified ? "QA OK" : "QA FLAG"}
              </span>
              <span className="pill">{minutes}m</span>
              <span className="pill">{hasAudio ? episode.tts_provider : "script only"}</span>
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
            {hasAudio ? (
              <PlayButton episode={episode} />
            ) : (
              <button className="button button-primary" disabled={audioBusy} onClick={generateAudio} type="button">
                {audioBusy ? "Generating audio..." : "Generate audio"}
              </button>
            )}
            <div style={{ display: "flex", gap: 8 }}>
              <a className="button button-secondary" href={episode.paper.pdf_url} rel="noreferrer" target="_blank">PDF</a>
              {hasAudio && episode.audio_url ? (
                <a className="button button-ghost" href={episode.audio_url} rel="noreferrer" target="_blank">Audio</a>
              ) : null}
            </div>
          </div>
        </div>

        <aside className="card column-stack detail-side">
          <span className="label">Abstract</span>
          <p className="abstract-body">{episode.paper.abstract}</p>
          <div className="divider" />
          <span className="label">Categories</span>
          <div className="tag-list">
            {episode.paper.categories.map((cat) => (<span className="pill" key={cat}>{cat}</span>))}
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
          ) : paragraphs.map((p, i) => <p key={i}>{p}</p>)}
        </div>
      </section>

      <DeepDiveChat episodeId={episode.id} />
    </div>
  )
}
