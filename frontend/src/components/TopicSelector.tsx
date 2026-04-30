'use client'

import { FormEvent, useEffect, useMemo, useRef, useState, useTransition } from "react"

import { saveTopics } from "@/lib/api"
import { emitToast } from "@/components/Toast"

type Props = {
  initialTopics: string[]
}

const STARTER_TOPICS = [
  "language models",
  "retrieval augmented generation",
  "robotics",
  "vision-language models",
  "on-device AI",
  "evaluation",
  "agents",
  "reasoning",
  "fine-tuning",
  "multimodal",
  "diffusion models",
  "speech",
  "code generation",
  "interpretability"
]

export function TopicSelector({ initialTopics }: Props) {
  const [topics, setTopics] = useState<string[]>(initialTopics)
  const [draft, setDraft] = useState("")
  const [isPending, startTransition] = useTransition()
  const [savedSnapshot, setSavedSnapshot] = useState<string[]>(initialTopics)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const suggestions = useMemo(
    () => STARTER_TOPICS.filter((topic) => !topics.includes(topic)),
    [topics]
  )

  const persist = (nextTopics: string[]) => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => {
      startTransition(async () => {
        try {
          const response = await saveTopics(nextTopics)
          setSavedSnapshot(response.topics)
        } catch (err) {
          emitToast(err instanceof Error ? err.message : "Save failed", "error")
        }
      })
    }, 380)
  }

  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current)
    }
  }, [])

  const addTopic = (value: string) => {
    const next = value.trim()
    if (!next || topics.includes(next)) return
    const updated = [...topics, next]
    setTopics(updated)
    setDraft("")
    persist(updated)
  }

  const removeTopic = (value: string) => {
    const updated = topics.filter((t) => t !== value)
    setTopics(updated)
    persist(updated)
  }

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const pendingDraft = draft.trim()
    if (!pendingDraft || topics.includes(pendingDraft)) return
    addTopic(pendingDraft)
  }

  const dirty = JSON.stringify(savedSnapshot) !== JSON.stringify(topics)

  return (
    <section className="card">
      <form onSubmit={submit}>
        <div className="row-between" style={{ marginBottom: 6 }}>
          <span className="label" style={{ margin: 0 }}>Add topic</span>
          <span className="meta-text" style={{ color: dirty || isPending ? "var(--accent)" : "var(--success)" }}>
            {isPending ? "saving..." : dirty ? "unsaved" : "saved ✓"}
          </span>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <input
            className="input"
            onChange={(event) => setDraft(event.target.value)}
            placeholder="e.g. efficient agents"
            value={draft}
          />
          <button className="button button-primary" type="submit">
            Add
          </button>
        </div>
      </form>

      <div className="divider" style={{ margin: "16px 0 12px" }} />

      <span className="label">Active ({topics.length})</span>
      {topics.length === 0 ? (
        <p style={{ fontSize: 12.5, color: "var(--muted)" }}>
          Add a topic to start ranking. Pick from the suggestions below or type your own.
        </p>
      ) : (
        <div className="filter-row">
          {topics.map((topic) => (
            <span className="chip chip-active chip-removable" key={topic}>
              {topic}
              <button
                className="chip-remove"
                onClick={() => removeTopic(topic)}
                type="button"
                aria-label={`Remove ${topic}`}
              >
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </span>
          ))}
        </div>
      )}

      {suggestions.length > 0 ? (
        <>
          <div className="divider" style={{ margin: "16px 0 12px" }} />
          <span className="label">Suggestions</span>
          <div className="filter-row">
            {suggestions.map((topic) => (
              <button className="chip" key={topic} onClick={() => addTopic(topic)} type="button">
                + {topic}
              </button>
            ))}
          </div>
        </>
      ) : null}
    </section>
  )
}
