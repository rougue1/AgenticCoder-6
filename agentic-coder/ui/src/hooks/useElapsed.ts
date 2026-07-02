import { useEffect, useState } from 'react'

// Seconds elapsed since `startedAtMs`, ticking once per second while `active`.
// When inactive it freezes at `frozenSeconds` (the server's final elapsed) so
// the timer stops cleanly on completion. Counts straight through pauses.
export function useElapsed(
  startedAtMs: number | null,
  active: boolean,
  frozenSeconds = 0,
): number {
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    if (!active) return
    const id = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(id)
  }, [active])

  if (!active) return frozenSeconds
  if (startedAtMs === null) return 0
  return Math.max(0, (now - startedAtMs) / 1000)
}
