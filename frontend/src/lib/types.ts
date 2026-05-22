export type Paper = {
  id: string
  arxiv_id: string
  title: string
  authors: string[]
  abstract: string
  categories: string[]
  published_at: string
  pdf_url: string
  citation_count: number
  score: number
}

export type Episode = {
  id: string
  title: string
  description: string
  topic: string
  score: number
  duration_secs: number
  tts_provider: string
  llm_provider: string
  qa_status: string
  created_at: string | null
  audio_ready: boolean
  audio_url: string | null
  script?: string | null
  paper: Paper
}

export type EpisodeListResponse = {
  items: Episode[]
  topics: string[]
}

export type TopicResponse = {
  topics: string[]
}

export type AskResponse = {
  answer: string
  citations: Array<{ section: string; excerpt: string }>
}

export type UserResponse = {
  id: string
  email: string
  feed_slug: string
  display_name?: string | null
}

export type AuthSession = {
  token: string
  user: UserResponse
}

export type StatusResponse = {
  auth_mode: string
  authenticated: boolean
  audio_backend: string
  generate_audio_on_pipeline: boolean
  live_discovery: boolean
  discovery_window_days: number
  scheduler_enabled: boolean
  user?: { feed_slug: string; email: string }
  topics?: string[]
  categories?: string[]
  providers: { llm: string; tts: string; embedder: string }
  last_job?: JobResponse | null
  provider_calls: Record<string, {
    ok: boolean
    at: string
    latency_ms?: number | null
    error?: string | null
    status?: number | null
  }>
}

export type JobResponse = {
  id: string
  status: "queued" | "running" | "done" | "error" | string
  window_days: number
  topics: string[]
  episode_count: number
  created_at: string | null
  started_at: string | null
  finished_at: string | null
  error: string | null
  result_count: number
  skipped_count: number
}
