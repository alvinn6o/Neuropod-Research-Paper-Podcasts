'use client'

import { useEffect, useState } from "react"

import { AuthGuard } from "@/components/AuthGuard"
import { ExploreView } from "@/components/ExploreView"
import { getEpisodes } from "@/lib/api"
import { Episode } from "@/lib/types"

export default function ExplorePage() {
  return (
    <AuthGuard>
      {() => <ExploreLoader />}
    </AuthGuard>
  )
}

function ExploreLoader() {
  const [items, setItems] = useState<Episode[]>([])
  const [topics, setTopics] = useState<string[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getEpisodes()
      .then((feed) => { setItems(feed.items); setTopics(feed.topics) })
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="card"><div className="skeleton" style={{ height: 80 }} /></div>
  return <ExploreView episodes={items} topics={topics} />
}
