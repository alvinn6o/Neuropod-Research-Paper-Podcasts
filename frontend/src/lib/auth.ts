/** Lightweight auth state for stub mode + Cognito-ready. */

const TOKEN_KEY = "neuropod-auth-token"

export type SessionUser = {
  id: string
  email: string
  feed_slug: string
  display_name?: string | null
  keys: Record<string, string>
}

export function getToken(): string | null {
  if (typeof window === "undefined") return null
  return window.localStorage.getItem(TOKEN_KEY)
}

export function setToken(token: string | null) {
  if (typeof window === "undefined") return
  if (token) window.localStorage.setItem(TOKEN_KEY, token)
  else window.localStorage.removeItem(TOKEN_KEY)
}

export function authHeaders(extra?: HeadersInit): HeadersInit {
  const token = getToken()
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...((extra as Record<string, string>) || {})
  }
  if (token) headers["Authorization"] = `Bearer ${token}`
  return headers
}
