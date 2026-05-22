'use client'

import { useEffect, useRef, useState, useTransition } from "react"

import { saveCategories } from "@/lib/api"
import { emitToast } from "@/components/Toast"

type Props = {
  initialCategories: string[]
}

// Curated list. Users can also paste custom ones (e.g. "hep-th").
const PRESETS: Array<{ id: string; label: string }> = [
  { id: "cs.AI", label: "AI" },
  { id: "cs.CL", label: "NLP" },
  { id: "cs.LG", label: "Machine Learning" },
  { id: "cs.CV", label: "Computer Vision" },
  { id: "cs.RO", label: "Robotics" },
  { id: "cs.IR", label: "Info Retrieval" },
  { id: "cs.NE", label: "Neural & Evol." },
  { id: "stat.ML", label: "Stat ML" },
  { id: "math.OC", label: "Optimization" },
  { id: "q-bio.QM", label: "Quant. Biology" },
  { id: "q-bio.NC", label: "Neuroscience" },
  { id: "q-fin.RM", label: "Risk Finance" },
  { id: "econ.GN", label: "Economics" },
  { id: "hep-th", label: "High-Energy Theory" },
  { id: "astro-ph.GA", label: "Astrophysics" },
]

export function CategorySelector({ initialCategories }: Props) {
  const [active, setActive] = useState<string[]>(initialCategories)
  const [draft, setDraft] = useState("")
  const [savedSnapshot, setSavedSnapshot] = useState<string[]>(initialCategories)
  const [isPending, startTransition] = useTransition()
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const persist = (next: string[]) => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => {
      startTransition(async () => {
        try {
          const response = await saveCategories(next)
          setSavedSnapshot(response.categories)
        } catch (err) {
          emitToast(err instanceof Error ? err.message : "Save failed", "error")
        }
      })
    }, 350)
  }

  useEffect(() => () => { if (debounceRef.current) clearTimeout(debounceRef.current) }, [])

  const toggle = (id: string) => {
    const next = active.includes(id) ? active.filter((c) => c !== id) : [...active, id]
    setActive(next)
    persist(next)
  }

  const addCustom = (raw: string) => {
    const clean = raw.trim()
    if (!clean || active.includes(clean)) return
    const next = [...active, clean]
    setActive(next)
    setDraft("")
    persist(next)
  }

  const dirty = JSON.stringify(savedSnapshot) !== JSON.stringify(active)

  return (
    <section className="card">
      <div className="row-between" style={{ marginBottom: 12 }}>
        <div>
          <span className="label" style={{ margin: 0 }}>arXiv categories <span style={{ color: "var(--muted)", fontWeight: 400, textTransform: "none" }}>(optional)</span></span>
          <p className="meta-text" style={{ marginTop: 4 }}>
            Pin papers to one or more arXiv categories. Leave empty to search across everything.
          </p>
        </div>
        <span className="meta-text" style={{ color: dirty || isPending ? "var(--accent)" : "var(--success)" }}>
          {isPending ? "saving..." : dirty ? "unsaved" : "saved ✓"}
        </span>
      </div>

      <div className="filter-row">
        {PRESETS.map((p) => {
          const isActive = active.includes(p.id)
          return (
            <button
              key={p.id}
              type="button"
              onClick={() => toggle(p.id)}
              className={isActive ? "chip chip-active" : "chip"}
              title={p.id}
            >
              {isActive ? "✓ " : "+ "}{p.label}
            </button>
          )
        })}
      </div>

      {active.length > 0 && active.some((c) => !PRESETS.find((p) => p.id === c)) ? (
        <>
          <div className="divider" style={{ margin: "16px 0 12px" }} />
          <span className="label">Custom</span>
          <div className="filter-row">
            {active.filter((c) => !PRESETS.find((p) => p.id === c)).map((c) => (
              <span className="chip chip-active chip-removable" key={c}>
                {c}
                <button className="chip-remove" onClick={() => toggle(c)} type="button" aria-label={`Remove ${c}`}>
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                    <line x1="18" y1="6" x2="6" y2="18" />
                    <line x1="6" y1="6" x2="18" y2="18" />
                  </svg>
                </button>
              </span>
            ))}
          </div>
        </>
      ) : null}

      <div className="divider" style={{ margin: "16px 0 12px" }} />
      <span className="label">Add custom category</span>
      <form
        onSubmit={(e) => { e.preventDefault(); addCustom(draft) }}
        style={{ display: "flex", gap: 8 }}
      >
        <input
          className="input"
          placeholder="e.g. hep-th, q-fin.PR"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
        />
        <button className="button button-secondary" type="submit" disabled={!draft.trim()}>
          Add
        </button>
      </form>
    </section>
  )
}
