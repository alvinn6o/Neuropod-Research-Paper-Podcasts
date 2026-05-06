'use client'

import { AuthGuard } from "@/components/AuthGuard"
import { FeedSubscribe } from "@/components/FeedSubscribe"
import { apiBaseUrl } from "@/lib/api"

export default function SubscribePage() {
  return (
    <AuthGuard>
      {(user) => <FeedSubscribe baseUrl={apiBaseUrl()} slug={user.feed_slug} />}
    </AuthGuard>
  )
}
