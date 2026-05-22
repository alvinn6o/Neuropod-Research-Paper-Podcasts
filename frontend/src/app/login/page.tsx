'use client'

import { FormEvent, useEffect, useState } from "react"
import { useRouter } from "next/navigation"

import { authMode, stubLogin } from "@/lib/api"
import { setToken } from "@/lib/auth"
import { emitToast } from "@/components/Toast"

export default function LoginPage() {
  const router = useRouter()
  const [email, setEmail] = useState("")
  const [mode, setMode] = useState<string>("stub")
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    authMode().then((r) => setMode(r.mode)).catch(() => undefined)
  }, [])

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    if (!email.trim() || busy) return
    setBusy(true)
    try {
      const session = await stubLogin(email.trim())
      setToken(session.token)
      router.replace("/")
    } catch (err) {
      emitToast(err instanceof Error ? err.message : "Login failed", "error")
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="stack-gap" style={{ maxWidth: 420, margin: "60px auto" }}>
      <section className="hero">
        <h1>Sign in</h1>
        <p>{mode === "stub"
          ? "Enter any email to create or resume a local session. No password required."
          : "Sign in with your Cognito account to continue."}</p>
      </section>

      {mode === "stub" ? (
        <form onSubmit={submit} className="card column-stack">
          <span className="label">Email</span>
          <input
            className="input"
            type="email"
            placeholder="you@example.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            autoFocus
          />
          <button className="button button-primary" type="submit" disabled={busy}>
            {busy ? "Signing in..." : "Continue"}
          </button>
        </form>
      ) : (
        <div className="card">
          <p className="meta-text">Cognito flow not wired in this build. Set NEUROPOD_AUTH_MODE=stub for local dev.</p>
        </div>
      )}
    </div>
  )
}
