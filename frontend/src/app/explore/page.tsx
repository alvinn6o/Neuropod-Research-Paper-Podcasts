import { ExploreView } from "@/components/ExploreView"
import { getEpisodes } from "@/lib/api"

export default async function ExplorePage() {
  const feed = await getEpisodes()
  return <ExploreView episodes={feed.items} topics={feed.topics} />
}
