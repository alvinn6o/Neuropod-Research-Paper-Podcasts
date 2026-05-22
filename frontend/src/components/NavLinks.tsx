'use client'

import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import { useEffect, useState } from "react"

import { ApiError, getMe } from "@/lib/api"
import { setToken } from "@/lib/auth"

const links = [
  { href: "/", label: "Feed" },
  { href: "/explore", label: "Explore" },
  { href: "/topics", label: "Topics" },
  { href: "/subscribe", label: "Feed URL" },
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
        <button className="nav-signout" type="button" onClick={signOut} title="Sign out">
          Sign out
        </button>
      ) : authed === false ? (
        <Link className={pathname === "/login" ? "active" : ""} href="/login">Sign in</Link>
      ) : null}
    </nav>
  )
}
