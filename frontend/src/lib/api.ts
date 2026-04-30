import { AskResponse, Episode, EpisodeListResponse, Paper, StatusResponse, TopicResponse } from "@/lib/types"

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "http://localhost:8000"

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    cache: "no-store",
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    }
  })

  if (!response.ok) {
    const message = await response.text()
    throw new Error(message || `Request failed for ${path}`)
  }

  return response.json() as Promise<T>
}

export function apiBaseUrl() {
  return API_BASE_URL
}

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
    body: JSON.stringify({ topics })
  })
}

export async function askEpisode(id: string, question: string) {
  return fetchJson<AskResponse>(`/episodes/${id}/ask`, {
    method: "POST",
    body: JSON.stringify({ question })
  })
}

export async function getStatus() {
  return fetchJson<StatusResponse>("/status")
}

export async function runPipeline(windowDays?: number) {
  const query = windowDays ? `?window=${windowDays}` : ""
  return fetchJson<{ queued: boolean; running: boolean; window_days?: number }>(`/pipeline/run${query}`, { method: "POST" })
}

export async function getPipelineState() {
  return fetchJson<{
    running: boolean
    started_at: string | null
    completed_at: string | null
    error: string | null
    last_count: number
  }>("/pipeline/state")
}

export async function postFeedback(episodeId: string, eventType: string, positionSecs = 0) {
  return fetchJson<{ ok: boolean; total_events: number }>("/feedback", {
    method: "POST",
    body: JSON.stringify({
      episode_id: episodeId,
      event_type: eventType,
      position_secs: Math.round(positionSecs)
    })
  })
}
