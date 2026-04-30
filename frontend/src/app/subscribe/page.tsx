import { FeedSubscribe } from "@/components/FeedSubscribe"
import { apiBaseUrl } from "@/lib/api"

export default function SubscribePage() {
  return (
    <div className="stack-gap">
      <section className="hero">
        <h1>Feed URL</h1>
        <p>Add to any podcast client supporting custom RSS.</p>
      </section>

      <FeedSubscribe baseUrl={apiBaseUrl()} />
    </div>
  )
}
