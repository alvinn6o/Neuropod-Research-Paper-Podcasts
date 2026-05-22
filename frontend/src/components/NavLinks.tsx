'use client'

import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import { useEffect, useState } from "react"

import { ApiError, getMe } from "@/lib/api"
import { setToken } from "@/lib/auth"

const links = [
  { href: "/", label: "Feed" },
  { href: "/topics", label: "Topics" },
]

export function NavLinks() {
  const pathname = usePathname()
  const router = useRouter()
  const [authed, setAuthed] = useState<boolean | null>(null)

  useEffect(() => {
    getMe()
      .then(() => setAuthed(true))
      .catch((err) => {
        if (err instanceof ApiError && err.status === 401) setAuthed(false)
        else setAuthed(false)
      })
  }, [pathname])

  const signOut = () => {
    setToken(null)
    router.replace("/login")
  }

  return (
    <nav className="nav-row">
      {links.map((link) => {
        const active = link.href === "/" ? pathname === "/" : pathname.startsWith(link.href)
        return (
          <Link className={active ? "active" : ""} href={link.href} key={link.href}>
            {link.label}
          </Link>
        )
      })}
      {authed ? (
        <>
          <Link
            className="nav-icon"
            href="/subscribe"
            title="RSS feed for podcast apps"
            aria-label="Your RSS feed URL"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M4 11a9 9 0 0 1 9 9" />
              <path d="M4 4a16 16 0 0 1 16 16" />
              <circle cx="5" cy="19" r="1" />
            </svg>
          </Link>
          <button className="nav-signout" type="button" onClick={signOut} title="Sign out">
            Sign out
          </button>
        </>
      ) : authed === false ? (
        <Link className={pathname === "/login" ? "active" : ""} href="/login">Sign in</Link>
      ) : null}
    </nav>
  )
}
