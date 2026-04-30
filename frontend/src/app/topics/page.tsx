import { TopicSelector } from "@/components/TopicSelector"
import { getTopics } from "@/lib/api"

export default async function TopicsPage() {
  const { topics } = await getTopics()

  return (
    <div className="stack-gap">
      <section className="hero">
        <h1>Topics</h1>
        <p>Used by the ranker to score new papers. Keep it tight.</p>
      </section>

      <TopicSelector initialTopics={topics} />
    </div>
  )
}
