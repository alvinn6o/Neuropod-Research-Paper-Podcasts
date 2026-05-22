'use client'

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"

import { ApiError, getMe } from "@/lib/api"
import { UserResponse } from "@/lib/types"

type Props = {
  children: (user: UserResponse) => React.ReactNode
}

export function AuthGuard({ children }: Props) {
  const router = useRouter()
  const [user, setUser] = useState<UserResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    getMe()
      .then((u) => { if (!cancelled) setUser(u) })
      .catch((err: unknown) => {
        if (cancelled) return
        if (err instanceof ApiError && err.status === 401) {
          router.replace("/login")
        } else {
          setError(err instanceof Error ? err.message : "Failed to load user")
        }
      })
    return () => { cancelled = true }
  }, [router])

  if (error) {
    return <div className="empty-state"><h3>Something went wrong</h3><p>{error}</p></div>
  }
  if (!user) {
    return (
      <div className="stack-gap">
        <div className="skeleton" style={{ height: 24, width: 180 }} />
        <div className="skeleton" style={{ height: 80, width: "100%" }} />
      </div>
    )
  }
  return <>{children(user)}</>
}
