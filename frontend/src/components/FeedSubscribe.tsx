'use client'

import { useEffect, useState } from "react"
import { emitToast } from "@/components/Toast"

type Props = {
  baseUrl: string
}

const STORAGE_KEY = "neuropod-feed-slug"

export function FeedSubscribe({ baseUrl }: Props) {
  const [slug, setSlug] = useState("demo")
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    const saved = window.localStorage.getItem(STORAGE_KEY)
    if (saved) setSlug(sanitize(saved))
  }, [])

  const update = (next: string) => {
    const cleaned = sanitize(next) || "demo"
    setSlug(cleaned)
    window.localStorage.setItem(STORAGE_KEY, cleaned)
  }

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
    <section className="card column-stack">
      <span className="label">Slug</span>
      <div style={{ display: "flex", gap: 8 }}>
        <input
          className="input"
          onChange={(e) => update(e.target.value)}
          placeholder="your-handle"
          value={slug}
        />
        <span className="meta-text" style={{ alignSelf: "center" }}>
          identifies your feed in podcast clients
        </span>
      </div>

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
        <a className="button button-secondary" href={feedUrl} rel="noreferrer" target="_blank">
          Browser
        </a>
        <a className="button button-secondary" href={`overcast://x-callback-url/add?url=${encodeURIComponent(feedUrl)}`}>
          Overcast
        </a>
        <a className="button button-secondary" href={`pcast://subscribe/${feedUrl.replace(/^https?:\/\//, "")}`}>
          Apple Podcasts
        </a>
        <a className="button button-secondary" href={`https://pca.st/itunes/${encodeURIComponent(feedUrl)}`}>
          Pocket Casts
        </a>
      </div>
    </section>
  )
}

function sanitize(input: string): string {
  return input.toLowerCase().replace(/[^a-z0-9-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40)
}
