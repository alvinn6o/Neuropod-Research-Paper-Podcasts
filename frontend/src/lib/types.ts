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
  qa_status: string
  created_at: string
  audio_url: string
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
  citations: Array<{
    section: string
    excerpt: string
  }>
}

export type StatusResponse = {
  demo_mode: boolean
  providers: {
    llm: string
    tts: string
    embedder: string
    openai: boolean
    anthropic: boolean
    elevenlabs: boolean
  }
  topics: string[]
  last_pipeline_run: string | null
  scheduler_enabled: boolean
  live_discovery: boolean
  discovery_window_days: number
  provider_calls: Record<string, {
    ok: boolean
    at: string
    latency_ms?: number | null
    error?: string | null
    status?: number | null
  }>
}
