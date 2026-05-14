'use client'

import { useState } from "react"
import { useRouter, useSearchParams } from "next/navigation"

import { deleteProviderKey, setProviderKey, setBedrockKey } from "@/lib/api"
import { UserResponse } from "@/lib/types"
import { emitToast } from "@/components/Toast"

const SIMPLE_PROVIDERS = [
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

const BEDROCK_REGIONS = ["us-east-1", "us-west-2", "us-east-2", "eu-west-1", "ap-northeast-1"]

type Props = { initialUser: UserResponse }

export function KeysForm({ initialUser }: Props) {
  const router = useRouter()
  const search = useSearchParams()
  const onboarding = search.get("onboarding") === "1"
  const [user, setUser] = useState<UserResponse>(initialUser)
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const [bedrock, setBedrock] = useState({
    region: "us-east-1",
    access_key: "",
    secret_key: "",
    session_token: "",
  })
  const [busy, setBusy] = useState<string | null>(null)

  const completeOnboardingIfDone = (next: UserResponse) => {
    if (onboarding && Object.keys(next.keys).length > 0) router.replace("/")
  }

  const save = async (provider: string) => {
    const value = (drafts[provider] || "").trim()
    if (!value) return
    setBusy(provider)
    try {
      const next = await setProviderKey(provider, value)
      setUser(next)
      setDrafts((prev) => ({ ...prev, [provider]: "" }))
      emitToast(`Saved ${provider} key`, "success")
      completeOnboardingIfDone(next)
    } catch (err) {
      emitToast(err instanceof Error ? err.message : "Save failed", "error")
    } finally {
      setBusy(null)
    }
  }

  const saveBedrock = async () => {
    if (!bedrock.region || !bedrock.access_key || !bedrock.secret_key) return
    setBusy("bedrock")
    try {
      const next = await setBedrockKey(bedrock)
      setUser(next)
      setBedrock({ region: bedrock.region, access_key: "", secret_key: "", session_token: "" })
      emitToast("Saved Bedrock credentials", "success")
      completeOnboardingIfDone(next)
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

      {SIMPLE_PROVIDERS.map((p) => {
        const masked = user.keys?.[p.id]
        const draft = drafts[p.id] ?? ""
        return (
          <div className="card column-stack" key={p.id}>
            <div className="row-between">
              <div>
                <span className="label" style={{ margin: 0 }}>{p.label}</span>
                <p className="meta-text" style={{ marginTop: 4 }}>{p.help}</p>
              </div>
              {masked ? <span className="metric">saved · …{masked}</span> : <span className="metric metric-warn">not set</span>}
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
                <button className="button button-ghost" type="button" onClick={() => remove(p.id)} disabled={busy === p.id}>
                  Remove
                </button>
              ) : null}
            </div>
          </div>
        )
      })}

      <div className="card column-stack">
        <div className="row-between">
          <div>
            <span className="label" style={{ margin: 0 }}>AWS Bedrock</span>
            <p className="meta-text" style={{ marginTop: 4 }}>
              Use Anthropic Claude through your own AWS account. Requires Bedrock model access enabled in the region.
            </p>
          </div>
          {user.keys?.bedrock ? (
            <span className="metric">saved · {user.keys.bedrock}</span>
          ) : (
            <span className="metric metric-warn">not set</span>
          )}
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: 8 }}>
          <select
            className="input"
            value={bedrock.region}
            onChange={(e) => setBedrock((prev) => ({ ...prev, region: e.target.value }))}
            disabled={busy === "bedrock"}
          >
            {BEDROCK_REGIONS.map((r) => (<option key={r} value={r}>{r}</option>))}
          </select>
          <input
            className="input"
            type="text"
            placeholder="AWS_ACCESS_KEY_ID"
            value={bedrock.access_key}
            onChange={(e) => setBedrock((prev) => ({ ...prev, access_key: e.target.value }))}
            disabled={busy === "bedrock"}
          />
        </div>
        <input
          className="input"
          type="password"
          placeholder="AWS_SECRET_ACCESS_KEY"
          value={bedrock.secret_key}
          onChange={(e) => setBedrock((prev) => ({ ...prev, secret_key: e.target.value }))}
          disabled={busy === "bedrock"}
        />
        <input
          className="input"
          type="password"
          placeholder="AWS_SESSION_TOKEN (optional, for STS)"
          value={bedrock.session_token}
          onChange={(e) => setBedrock((prev) => ({ ...prev, session_token: e.target.value }))}
          disabled={busy === "bedrock"}
        />
        <div style={{ display: "flex", gap: 8 }}>
          <button
            className="button button-primary"
            type="button"
            onClick={saveBedrock}
            disabled={busy === "bedrock" || !bedrock.access_key || !bedrock.secret_key}
          >
            {busy === "bedrock" ? "Saving…" : "Save Bedrock"}
          </button>
          {user.keys?.bedrock ? (
            <button className="button button-ghost" type="button" onClick={() => remove("bedrock")} disabled={busy === "bedrock"}>
              Remove
            </button>
          ) : null}
        </div>
        <p className="meta-text">
          Need an IAM user? Attach <code className="kbd">AmazonBedrockFullAccess</code> (or scope down to <code className="kbd">bedrock:InvokeModel</code>). Then enable Anthropic Claude model access in the Bedrock console for your region.
        </p>
      </div>
    </div>
  )
}
