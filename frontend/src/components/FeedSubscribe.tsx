'use client'

import { useState } from "react"
import { emitToast } from "@/components/Toast"

type Props = {
  baseUrl: string
  slug: string
}

export function FeedSubscribe({ baseUrl, slug }: Props) {
  const [copied, setCopied] = useState(false)
  const feedUrl = `${baseUrl}/feed/${encodeURIComponent(slug)}`

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(feedUrl)
      setCopied(true)
      emitToast("URL copied", "success")
      setTimeout(() => setCopied(false), 1500)
    } catch {
      emitToast("Copy failed", "error")
    }
  }

  return (
    <div className="stack-gap">
      <section className="hero">
        <h1>Feed URL</h1>
        <p>Add to any podcast client supporting custom RSS.</p>
      </section>

      <section className="card column-stack">
        <span className="label">Your slug</span>
        <p className="meta-text">Set when you signed up. Stable, used in your feed URL.</p>
        <div className="subscribe-url">{slug}</div>

        <div className="divider" />

        <span className="label">RSS endpoint</span>
        <div style={{ display: "flex", gap: 8 }}>
          <div className="subscribe-url" style={{ flex: 1 }}>{feedUrl}</div>
          <button className="button button-secondary" onClick={copy} type="button">
            {copied ? "Copied" : "Copy"}
          </button>
        </div>

        <div className="divider" />

        <span className="label">Open in</span>
        <div className="filter-row">
          <a className="button button-secondary" href={feedUrl} rel="noreferrer" target="_blank">Browser</a>
          <a className="button button-secondary" href={`overcast://x-callback-url/add?url=${encodeURIComponent(feedUrl)}`}>Overcast</a>
          <a className="button button-secondary" href={`pcast://subscribe/${feedUrl.replace(/^https?:\/\//, "")}`}>Apple Podcasts</a>
          <a className="button button-secondary" href={`https://pca.st/itunes/${encodeURIComponent(feedUrl)}`}>Pocket Casts</a>
        </div>
      </section>
    </div>
  )
}
