'use client'

import { FormEvent, useEffect, useRef, useState, useTransition } from "react"

import { askEpisode } from "@/lib/api"
import { AskResponse } from "@/lib/types"
import { emitToast } from "@/components/Toast"

type Props = {
  episodeId: string
}

type Turn = {
  id: string
  question: string
  answer?: string
  citations?: AskResponse["citations"]
  pending?: boolean
  error?: string
}

export function DeepDiveChat({ episodeId }: Props) {
  const [question, setQuestion] = useState("")
  const [turns, setTurns] = useState<Turn[]>([])
  const [isPending, startTransition] = useTransition()
  const endRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" })
  }, [turns])

  const submit = (event: FormEvent) => {
    event.preventDefault()
    const trimmed = question.trim()
    if (trimmed.length < 3) return

    const turnId = Math.random().toString(36).slice(2, 9)
    setTurns((prev) => [...prev, { id: turnId, question: trimmed, pending: true }])
    setQuestion("")

    startTransition(async () => {
      try {
        const next = await askEpisode(episodeId, trimmed)
        setTurns((prev) =>
          prev.map((t) => (t.id === turnId ? { ...t, answer: next.answer, citations: next.citations, pending: false } : t))
        )
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Question failed"
        setTurns((prev) => prev.map((t) => (t.id === turnId ? { ...t, error: msg, pending: false } : t)))
        emitToast(msg, "error")
      }
    })
  }

  const clear = () => setTurns([])

  return (
    <section className="card">
      <div className="row-between" style={{ marginBottom: 12 }}>
        <span className="label" style={{ margin: 0 }}>Ask the paper</span>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <span className="meta-text">Grounded in retrieved chunks</span>
          {turns.length > 0 ? (
            <button className="button button-ghost" onClick={clear} type="button" style={{ height: 24, padding: "0 8px", fontSize: 11 }}>
              Clear
            </button>
          ) : null}
        </div>
      </div>

      {turns.length > 0 ? (
        <div className="chat-thread">
          {turns.map((turn) => (
            <div className="chat-turn" key={turn.id}>
              <div className="chat-question">
                <span className="chat-prompt">›</span>
                <span>{turn.question}</span>
              </div>
              {turn.pending ? (
                <div className="chat-pending">
                  <div className="skeleton" style={{ height: 12, width: "92%" }} />
                  <div className="skeleton" style={{ height: 12, width: "78%" }} />
                  <div className="skeleton" style={{ height: 12, width: "84%" }} />
                </div>
              ) : turn.error ? (
                <p className="status-line error">{turn.error}</p>
              ) : (
                <div className="chat-answer">
                  <p>{turn.answer}</p>
                  {turn.citations && turn.citations.length > 0 ? (
                    <div className="citation-list" style={{ marginTop: 8 }}>
                      {turn.citations.map((citation, index) => (
                        <div className="citation" key={`${citation.section}-${index}`}>
                          <strong>{citation.section}</strong>
                          <span>{citation.excerpt}</span>
                        </div>
                      ))}
                    </div>
                  ) : null}
                </div>
              )}
            </div>
          ))}
          <div ref={endRef} />
        </div>
      ) : null}

      <form onSubmit={submit} style={{ marginTop: turns.length ? 12 : 0 }}>
        <textarea
          className="textarea"
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
              e.preventDefault()
              submit(e as unknown as FormEvent)
            }
          }}
          placeholder={turns.length ? "Ask a follow-up..." : "Ask a specific question about methodology, results, or limitations..."}
          rows={2}
          value={question}
        />
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 10 }}>
          <span className="meta-text"><span className="kbd">⌘</span> <span className="kbd">↵</span> to send</span>
          <button className="button button-primary" disabled={isPending || question.trim().length < 3} type="submit">
            {isPending ? "Retrieving..." : "Ask"}
          </button>
        </div>
      </form>
    </section>
  )
}
