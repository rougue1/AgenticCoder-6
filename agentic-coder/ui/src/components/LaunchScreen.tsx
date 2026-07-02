import { useEffect, useState } from 'react'
import { Hammer, Loader2, Play, RotateCcw } from 'lucide-react'
import { usePipelineStore } from '../store/pipelineStore'
import { fetchState, resumeBuild, startBuild } from '../api/client'

// View A — full-viewport launch screen shown while the pipeline is idle.
export function LaunchScreen() {
  const markStarting = usePipelineStore((s) => s.markStarting)
  const applyState = usePipelineStore((s) => s.applyState)
  const connected = usePipelineStore((s) => s.connected)

  const [prompt, setPrompt] = useState('')
  const [projectDir, setProjectDir] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [resumable, setResumable] = useState<null | string>(null) // project name if resumable

  // Detect an existing, resumable on-disk project (status paused | blocked).
  useEffect(() => {
    let alive = true
    fetchState()
      .then((s) => {
        if (!alive) return
        if (s.status === 'paused' || s.status === 'blocked') {
          setResumable(s.project_name || s.project_dir || 'previous build')
        }
      })
      .catch(() => {})
    return () => {
      alive = false
    }
  }, [])

  const start = async () => {
    if (!prompt.trim() || busy) return
    setBusy(true)
    setError('')
    try {
      await startBuild(prompt.trim(), projectDir.trim() || undefined)
      markStarting() // transition immediately to View B
    } catch (e: any) {
      setError(e?.message || 'failed to start')
      setBusy(false)
    }
  }

  const resume = async () => {
    setBusy(true)
    setError('')
    try {
      await resumeBuild(projectDir.trim() || undefined)
      const s = await fetchState()
      applyState({ ...s, status: 'running', running: true })
    } catch (e: any) {
      setError(e?.message || 'failed to resume')
      setBusy(false)
    }
  }

  return (
    <div className="flex h-screen w-screen flex-col items-center justify-center bg-vs-bg px-6">
      <div className="w-full max-w-2xl">
        {/* Wordmark */}
        <div className="mb-8 flex items-center justify-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-lg bg-vs-accent/20 ring-1 ring-vs-accent/50">
            <Hammer className="h-6 w-6 text-vs-blue" />
          </div>
          <div>
            <h1 className="text-3xl font-semibold tracking-tight text-vs-text">
              AI<span className="text-vs-blue">Forge</span>
            </h1>
            <p className="text-xs text-vs-muted">local autonomous coding pipeline</p>
          </div>
        </div>

        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') start()
          }}
          placeholder="Describe the application you want to build..."
          rows={6}
          className="w-full resize-none rounded-md border border-vs-border bg-vs-panel p-4 text-[15px] text-vs-text placeholder-vs-muted shadow-inner outline-none focus:border-vs-accent focus:ring-1 focus:ring-vs-accent"
        />

        <input
          value={projectDir}
          onChange={(e) => setProjectDir(e.target.value)}
          placeholder="Project directory (leave blank for default sandbox)"
          className="mt-3 w-full rounded-md border border-vs-border bg-vs-panel px-4 py-2.5 text-sm text-vs-text placeholder-vs-muted outline-none focus:border-vs-accent focus:ring-1 focus:ring-vs-accent"
        />

        {error && (
          <div className="mt-3 rounded-md border border-vs-red/40 bg-vs-red/10 px-3 py-2 text-sm text-vs-red">
            {error}
          </div>
        )}

        <div className="mt-5 flex items-center gap-3">
          <button
            onClick={start}
            disabled={!prompt.trim() || busy}
            className="inline-flex items-center gap-2 rounded-md bg-vs-accent px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-vs-accent-hover disabled:cursor-not-allowed disabled:opacity-40"
          >
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            Start Build
          </button>

          {resumable && (
            <button
              onClick={resume}
              disabled={busy}
              className="inline-flex items-center gap-2 rounded-md border border-vs-border bg-vs-panel px-4 py-2.5 text-sm text-vs-text transition-colors hover:border-vs-accent hover:text-white disabled:opacity-40"
            >
              <RotateCcw className="h-4 w-4" />
              Resume Previous Build
              <span className="text-vs-muted">· {resumable}</span>
            </button>
          )}

          <div className="ml-auto flex items-center gap-2 text-xs text-vs-muted">
            <span
              className={`h-2 w-2 rounded-full ${connected ? 'bg-vs-green' : 'bg-vs-red'}`}
            />
            {connected ? 'server connected' : 'connecting…'}
          </div>
        </div>

        <p className="mt-6 text-center text-xs text-vs-muted">
          Tip: press <kbd className="rounded bg-vs-panel2 px-1.5 py-0.5">⌘/Ctrl</kbd>+
          <kbd className="rounded bg-vs-panel2 px-1.5 py-0.5">Enter</kbd> to start
        </p>
      </div>
    </div>
  )
}
