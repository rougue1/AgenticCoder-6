import { useEffect, useRef } from 'react'
import { Terminal } from 'xterm'
import { FitAddon } from 'xterm-addon-fit'

// Read-only xterm.js block for command stdout/stderr (ANSI colour support,
// black background, monospace). Mounted only while its row is expanded so the
// fit addon always has real dimensions to measure.
export function XTerm({ text }: { text: string }) {
  const hostRef = useRef<HTMLDivElement | null>(null)
  const termRef = useRef<Terminal | null>(null)
  const fitRef = useRef<FitAddon | null>(null)

  useEffect(() => {
    if (!hostRef.current) return
    const term = new Terminal({
      convertEol: true,
      disableStdin: true,
      cursorBlink: false,
      fontSize: 12,
      fontFamily: 'JetBrains Mono, Menlo, Consolas, monospace',
      scrollback: 2000,
      theme: {
        background: '#0c0c0c',
        foreground: '#cccccc',
        selectionBackground: '#264f78',
      },
    })
    const fit = new FitAddon()
    term.loadAddon(fit)
    term.open(hostRef.current)
    try {
      fit.fit()
    } catch {
      /* container not laid out yet */
    }
    term.write(text || '(no output)')
    termRef.current = term
    fitRef.current = fit

    const ro = new ResizeObserver(() => {
      try {
        fit.fit()
      } catch {
        /* ignore */
      }
    })
    ro.observe(hostRef.current)
    return () => {
      ro.disconnect()
      term.dispose()
      termRef.current = null
      fitRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Re-render when the captured text changes (e.g. a re-run of the same command).
  useEffect(() => {
    const term = termRef.current
    if (!term) return
    term.clear()
    term.write(text || '(no output)')
  }, [text])

  return <div ref={hostRef} className="h-full w-full" />
}
