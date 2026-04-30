import type { Metadata } from "next"
import Link from "next/link"

import { AudioPlayer } from "@/components/AudioPlayer"
import { ToastProvider } from "@/components/Toast"
import { NavLinks } from "@/components/NavLinks"
import { StatusBadge } from "@/components/StatusBadge"
import { CommandPalette } from "@/components/CommandPalette"
import { CommandTrigger } from "@/components/CommandTrigger"
import { ShortcutsHint } from "@/components/ShortcutsHint"

import "./globals.css"

export const metadata: Metadata = {
  title: "Neuropod",
  description: "Research papers, distilled into audio.",
  manifest: "/manifest.json",
  icons: {
    icon: [{ url: "/icon.svg", type: "image/svg+xml" }]
  }
}

export default async function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body>
        <ToastProvider>
          <div className="shell">
            <header className="topbar">
              <Link className="brand-lockup" href="/">
                <div className="brand-mark">N</div>
                <span className="brand">Neuropod</span>
                <span className="brand-suffix">AI</span>
              </Link>
              <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <NavLinks />
                <CommandTrigger />
                <StatusBadge />
              </div>
            </header>
            <main className="page-shell">{children}</main>
          </div>
          <AudioPlayer />
          <CommandPalette />
          <ShortcutsHint />
        </ToastProvider>
      </body>
    </html>
  )
}
