/** Split script into 3-5 sentence paragraphs for readability. */
export function splitScript(script: string): string[] {
  if (!script) return []

  const existing = script.split(/\n{2,}/).map((p) => p.trim()).filter(Boolean)
  if (existing.length > 1) return existing

  const sentences = script
    .replace(/\s+/g, " ")
    .trim()
    .split(/(?<=[.!?])\s+(?=[A-Z])/)
    .filter((s) => s.length > 0)

  if (sentences.length <= 4) return [sentences.join(" ")]

  const groupSize = Math.max(3, Math.ceil(sentences.length / 4))
  const paragraphs: string[] = []
  for (let i = 0; i < sentences.length; i += groupSize) {
    paragraphs.push(sentences.slice(i, i + groupSize).join(" "))
  }
  return paragraphs
}
