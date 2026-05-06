'use client'

import { KeysForm } from "@/components/KeysForm"
import { AuthGuard } from "@/components/AuthGuard"

export default function KeysSettingsPage() {
  return (
    <AuthGuard>
      {(user) => (
        <div className="stack-gap" style={{ maxWidth: 720, margin: "0 auto" }}>
          <section className="hero">
            <h1>API keys</h1>
            <p>Bring your own provider keys. Stored encrypted, used only for your requests.</p>
          </section>
          <KeysForm initialUser={user} />
        </div>
      )}
    </AuthGuard>
  )
}
