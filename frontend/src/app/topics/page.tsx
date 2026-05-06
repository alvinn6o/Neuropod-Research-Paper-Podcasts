'use client'

import { useEffect, useState } from "react"

import { AuthGuard } from "@/components/AuthGuard"
import { TopicSelector } from "@/components/TopicSelector"
import { getTopics } from "@/lib/api"

export default function TopicsPage() {
  return (
    <AuthGuard>
      {() => <TopicsView />}
    </AuthGuard>
  )
}

function TopicsView() {
  const [topics, setTopics] = useState<string[] | null>(null)

  useEffect(() => {
    getTopics().then((r) => setTopics(r.topics)).catch(() => setTopics([]))
  }, [])

  return (
    <div className="stack-gap">
      <section className="hero">
        <h1>Topics</h1>
        <p>Used by the ranker to score new papers. Keep it tight.</p>
      </section>
      {topics === null ? (
        <div className="card"><div className="skeleton" style={{ height: 80 }} /></div>
      ) : (
        <TopicSelector initialTopics={topics} />
      )}
    </div>
  )
}
