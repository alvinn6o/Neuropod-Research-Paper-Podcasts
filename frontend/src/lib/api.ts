import { authHeaders, setToken } from "@/lib/auth"
import {
  AskResponse,
  AuthSession,
  Episode,
  EpisodeListResponse,
  JobResponse,
  Paper,
  StatusResponse,
  TopicResponse,
  UserResponse,
} from "@/lib/types"

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "http://localhost:8000"

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
    this.name = "ApiError"
  }
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    cache: "no-store",
    headers: authHeaders(init?.headers),
  })

  if (!response.ok) {
    let message = ""
    try { message = (await response.json())?.detail ?? "" } catch { message = await response.text() }
    if (response.status === 401) {
      setToken(null)
      if (typeof window !== "undefined" && !window.location.pathname.startsWith("/login")) {
        window.location.assign("/login")
      }
    }
    throw new ApiError(response.status, message || `Request failed for ${path}`)
  }

  if (response.status === 204) return null as T
  return response.json() as Promise<T>
}

export function apiBaseUrl() {
  return API_BASE_URL
}

// ---- Auth ----
export async function authMode() {
  return fetchJson<{ mode: string }>("/auth/mode")
}

export async function stubLogin(email: string) {
  return fetchJson<AuthSession>("/auth/stub/login", {
    method: "POST",
    body: JSON.stringify({ email }),
  })
}

// ---- Me ----
export async function getMe() {
  return fetchJson<UserResponse>("/me")
}

export async function setProviderKey(provider: string, apiKey: string) {
  return fetchJson<UserResponse>("/me/keys", {
    method: "PUT",
    body: JSON.stringify({ provider, api_key: apiKey }),
  })
}

export type BedrockCredentials = {
  region: string
  access_key: string
  secret_key: string
  session_token?: string
  model_id?: string
}

export async function setBedrockKey(creds: BedrockCredentials) {
  return fetchJson<UserResponse>("/me/keys", {
    method: "PUT",
    body: JSON.stringify({ provider: "bedrock", ...creds }),
  })
}

export async function deleteProviderKey(provider: string) {
  return fetchJson<UserResponse>(`/me/keys/${provider}`, { method: "DELETE" })
}

// ---- Domain ----
export async function getEpisodes(topic?: string) {
  const query = topic ? `?topic=${encodeURIComponent(topic)}` : ""
  return fetchJson<EpisodeListResponse>(`/episodes${query}`)
}

export async function getEpisode(id: string) {
  return fetchJson<Episode>(`/episodes/${id}`)
}

export async function getEpisodePaper(id: string) {
  return fetchJson<Paper>(`/episodes/${id}/paper`)
}

export async function getTopics() {
  return fetchJson<TopicResponse>("/topics")
}

export async function saveTopics(topics: string[]) {
  return fetchJson<TopicResponse>("/topics", {
    method: "POST",
    body: JSON.stringify({ topics }),
  })
}

export async function askEpisode(id: string, question: string) {
  return fetchJson<AskResponse>(`/episodes/${id}/ask`, {
    method: "POST",
    body: JSON.stringify({ question }),
  })
}

export async function getStatus() {
  return fetchJson<StatusResponse>("/status")
}

export async function postFeedback(episodeId: string, eventType: string, positionSecs = 0) {
  return fetchJson<{ ok: boolean; total_events: number }>("/feedback", {
    method: "POST",
    body: JSON.stringify({
      episode_id: episodeId,
      event_type: eventType,
      position_secs: Math.round(positionSecs),
    }),
  })
}

export async function runPipeline(windowDays?: number) {
  const query = windowDays ? `?window=${windowDays}` : ""
  return fetchJson<JobResponse>(`/pipeline/run${query}`, { method: "POST" })
}

export async function getPipelineState() {
  return fetchJson<JobResponse | null>("/pipeline/state")
}

export async function getPipelineJob(jobId: string) {
  return fetchJson<JobResponse>(`/pipeline/jobs/${jobId}`)
}
