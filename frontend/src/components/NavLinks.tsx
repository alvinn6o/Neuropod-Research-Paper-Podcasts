'use client'

import Link from "next/link"
import { usePathname } from "next/navigation"

const links = [
  { href: "/", label: "Feed" },
  { href: "/explore", label: "Explore" },
  { href: "/topics", label: "Topics" },
  { href: "/subscribe", label: "Feed URL" }
]

export function NavLinks() {
  const pathname = usePathname()

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
    </nav>
  )
}
