'use client'

import { useState } from "react"
import { useRouter, useSearchParams } from "next/navigation"

import { deleteProviderKey, setProviderKey } from "@/lib/api"
import { UserResponse } from "@/lib/types"
import { emitToast } from "@/components/Toast"

const PROVIDERS = [
  {
    id: "anthropic",
    label: "Anthropic",
    placeholder: "sk-ant-...",
    help: "Used for script generation. Get one at console.anthropic.com",
  },
  {
    id: "openai",
    label: "OpenAI",
    placeholder: "sk-...",
    help: "Used for fallback script generation, embeddings, and TTS. platform.openai.com",
  },
  {
    id: "elevenlabs",
    label: "ElevenLabs",
    placeholder: "...",
    help: "Used for high-quality TTS. elevenlabs.io",
  },
] as const

type Props = { initialUser: UserResponse }

export function KeysForm({ initialUser }: Props) {
  const router = useRouter()
  const search = useSearchParams()
  const onboarding = search.get("onboarding") === "1"
  const [user, setUser] = useState<UserResponse>(initialUser)
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const [busy, setBusy] = useState<string | null>(null)

  const save = async (provider: string) => {
    const value = (drafts[provider] || "").trim()
    if (!value) return
    setBusy(provider)
    try {
      const next = await setProviderKey(provider, value)
      setUser(next)
      setDrafts((prev) => ({ ...prev, [provider]: "" }))
      emitToast(`Saved ${provider} key`, "success")
      if (onboarding && Object.keys(next.keys).length > 0) {
        router.replace("/")
      }
    } catch (err) {
      emitToast(err instanceof Error ? err.message : "Save failed", "error")
    } finally {
      setBusy(null)
    }
  }

  const remove = async (provider: string) => {
    setBusy(provider)
    try {
      const next = await deleteProviderKey(provider)
      setUser(next)
      emitToast(`Removed ${provider} key`, "success")
    } catch (err) {
      emitToast(err instanceof Error ? err.message : "Delete failed", "error")
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="stack-gap">
      {onboarding ? (
        <div className="card" style={{ borderColor: "var(--accent-line)" }}>
          <span className="label" style={{ color: "var(--accent)" }}>One thing</span>
          <p style={{ marginTop: 6 }}>Add at least one provider key to start generating episodes. Keys live in your account, encrypted, and only your requests use them.</p>
        </div>
      ) : null}

      {PROVIDERS.map((p) => {
        const masked = user.keys?.[p.id]
        const draft = drafts[p.id] ?? ""
        return (
          <div className="card column-stack" key={p.id}>
            <div className="row-between">
              <div>
                <span className="label" style={{ margin: 0 }}>{p.label}</span>
                <p className="meta-text" style={{ marginTop: 4 }}>{p.help}</p>
              </div>
              {masked ? (
                <span className="metric">saved · …{masked}</span>
              ) : (
                <span className="metric metric-warn">not set</span>
              )}
            </div>

            <div style={{ display: "flex", gap: 8 }}>
              <input
                className="input"
                type="password"
                placeholder={p.placeholder}
                value={draft}
                onChange={(e) => setDrafts((prev) => ({ ...prev, [p.id]: e.target.value }))}
                disabled={busy === p.id}
              />
              <button
                className="button button-primary"
                type="button"
                onClick={() => save(p.id)}
                disabled={busy === p.id || !draft.trim()}
              >
                {busy === p.id ? "Saving…" : "Save"}
              </button>
              {masked ? (
                <button
                  className="button button-ghost"
                  type="button"
                  onClick={() => remove(p.id)}
                  disabled={busy === p.id}
                >
                  Remove
                </button>
              ) : null}
            </div>
          </div>
        )
      })}
    </div>
  )
}
