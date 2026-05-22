'use client'

import { useEffect, useState } from "react"

import { AuthGuard } from "@/components/AuthGuard"
import { CategorySelector } from "@/components/CategorySelector"
import { TopicSelector } from "@/components/TopicSelector"
import { getCategories, getTopics } from "@/lib/api"

export default function TopicsPage() {
  return (
    <AuthGuard>
      {() => <TopicsView />}
    </AuthGuard>
  )
}

function TopicsView() {
  const [topics, setTopics] = useState<string[] | null>(null)
  const [categories, setCategories] = useState<string[] | null>(null)

  useEffect(() => {
    getTopics().then((r) => setTopics(r.topics)).catch(() => setTopics([]))
    getCategories().then((r) => setCategories(r.categories)).catch(() => setCategories([]))
  }, [])

  const ready = topics !== null && categories !== null

  return (
    <div className="stack-gap">
      <section className="hero">
        <h1>Topics &amp; categories</h1>
        <p>Topics drive the semantic ranker. arXiv categories optionally narrow the corpus.</p>
      </section>

      {!ready ? (
        <div className="card"><div className="skeleton" style={{ height: 80 }} /></div>
      ) : (
        <>
          <TopicSelector initialTopics={topics!} />
          <CategorySelector initialCategories={categories!} />
        </>
      )}
    </div>
  )
}
