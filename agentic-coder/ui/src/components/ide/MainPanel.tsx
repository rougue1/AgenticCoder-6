import { useEffect, useMemo, useRef, useState } from 'react'
import { CheckCircle2, FileCode2, Loader2, X } from 'lucide-react'
import { usePipelineStore } from '../../store/pipelineStore'
import { CodeView } from '../common/CodeView'
import { basename } from '../../lib/format'
import { fetchFile } from '../../api/client'

export function MainPanel() {
  const openFiles = usePipelineStore((s) => s.openFiles)
  const activeFile = usePipelineStore((s) => s.activeFile)
  const writingPath = usePipelineStore((s) => s.writingPath)
  const setActiveFile = usePipelineStore((s) => s.setActiveFile)
  const closeFile = usePipelineStore((s) => s.closeFile)
  const status = usePipelineStore((s) => s.status)

  const active = openFiles.find((f) => f.path === activeFile) || null

  return (
    <section className="flex h-full min-w-0 flex-col bg-vs-bg">
      {/* tab bar */}
      {openFiles.length > 0 && (
        <div className="flex h-9 shrink-0 items-stretch overflow-x-auto border-b border-vs-border bg-vs-panel">
          {openFiles.map((f) => {
            const isActive = f.path === activeFile
            const isWriting = f.path === writingPath
            return (
              <div
                key={f.path}
                onClick={() => setActiveFile(f.path)}
                className={`group flex cursor-pointer items-center gap-2 border-r border-vs-border px-3 text-[13px] ${
                  isActive ? 'bg-vs-bg text-vs-text' : 'bg-vs-tab text-vs-text-dim hover:text-vs-text'
                }`}
                title={f.path}
              >
                <FileCode2 className="h-3.5 w-3.5 shrink-0 text-vs-blue/70" />
                <span className="max-w-[160px] truncate">{basename(f.path)}</span>
                {isWriting && (
                  <span className="flex items-center gap-1 rounded bg-vs-yellow/15 px-1.5 py-0.5 text-[10px] text-vs-yellow">
                    <Loader2 className="h-2.5 w-2.5 animate-spin" /> Writing…
                  </span>
                )}
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    closeFile(f.path)
                  }}
                  className="ml-1 rounded p-0.5 text-vs-muted opacity-0 hover:bg-vs-panel2 hover:text-vs-text group-hover:opacity-100"
                >
                  <X className="h-3 w-3" />
                </button>
              </div>
            )
          })}
        </div>
      )}

      {/* content */}
      <div className="min-h-0 flex-1 overflow-auto">
        {active ? (
          <FileView path={active.path} content={active.content} />
        ) : status === 'done' || status === 'blocked' || status === 'error' || status === 'cancelled' ? (
          <CompletionSummary />
        ) : (
          <StatusCard />
        )}
      </div>
    </section>
  )
}

// ── active file (with streaming reveal while being written) ───────────────────
function FileView({ path, content }: { path: string; content: string }) {
  const writingPath = usePipelineStore((s) => s.writingPath)
  const writtenTs = usePipelineStore((s) => s.filesWritten[path])
  const endWriting = usePipelineStore((s) => s.endWriting)
  const isWriting = writingPath === path

  const [reveal, setReveal] = useState(content.length)
  const raf = useRef<number | null>(null)

  useEffect(() => {
    if (!isWriting) {
      setReveal(content.length)
      return
    }
    // Animate the file in over ~1s in ~40 chunks (bounded re-highlights).
    const total = content.length
    const steps = 40
    const chunk = Math.max(1, Math.ceil(total / steps))
    let cur = 0
    setReveal(0)
    const tick = () => {
      cur = Math.min(total, cur + chunk)
      setReveal(cur)
      if (cur < total) {
        raf.current = window.setTimeout(tick, 1000 / steps)
      } else {
        endWriting(path)
      }
    }
    raf.current = window.setTimeout(tick, 1000 / steps)
    return () => {
      if (raf.current) window.clearTimeout(raf.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, isWriting, writtenTs, content])

  const shown = isWriting ? content.slice(0, reveal) : content
  return <CodeView path={path} content={shown} showCursor={isWriting && reveal < content.length} />
}

// ── running, no file open ─────────────────────────────────────────────────────
function StatusCard() {
  const state = usePipelineStore((s) => s.state)
  const thinking = usePipelineStore((s) => s.thinking)
  const currentId = usePipelineStore((s) => s.currentBlockId)
  const current = thinking.find((b) => b.id === currentId)
  const summary = current?.output?.slice(-1600) || ''

  return (
    <div className="mx-auto flex h-full max-w-3xl flex-col items-center justify-center gap-4 px-8 text-center">
      <div className="flex items-center gap-2 text-vs-blue">
        <Loader2 className="h-5 w-5 animate-spin" />
        <span className="text-sm uppercase tracking-widest">{state.phase || 'starting'}</span>
      </div>
      {state.current_subtask ? (
        <>
          <h2 className="text-xl font-medium text-vs-text">{state.current_subtask}</h2>
          {state.current_subtask_intent && (
            <p className="max-w-xl text-sm leading-relaxed text-vs-text-dim">
              {state.current_subtask_intent}
            </p>
          )}
        </>
      ) : (
        <h2 className="text-xl font-medium text-vs-text">
          {state.project_name || 'Building your project…'}
        </h2>
      )}
      {summary && (
        <pre className="mt-2 max-h-64 w-full overflow-auto whitespace-pre-wrap rounded-md border border-vs-border bg-vs-panel p-3 text-left font-mono text-xs text-vs-text-dim">
          {summary}
        </pre>
      )}
      <p className="text-xs text-vs-muted">
        Select a file on the left to view it, or watch it stream as the model writes.
      </p>
    </div>
  )
}

// ── terminal state ────────────────────────────────────────────────────────────
function CompletionSummary() {
  const state = usePipelineStore((s) => s.state)
  const status = usePipelineStore((s) => s.status)
  const startedAtMs = usePipelineStore((s) => s.startedAtMs)
  const [blocked, setBlocked] = useState('')

  const totalTime = useMemo(() => {
    if (state.elapsed_seconds) return state.elapsed_seconds
    if (startedAtMs) return (Date.now() - startedAtMs) / 1000
    return 0
  }, [state.elapsed_seconds, startedAtMs])

  useEffect(() => {
    fetchFile('.agent/blocked.md')
      .then((r) => setBlocked(r.exists ? r.content : ''))
      .catch(() => {})
  }, [])

  const headline =
    status === 'done'
      ? 'Build complete'
      : status === 'blocked'
        ? 'Build finished with blocked work'
        : status === 'cancelled'
          ? 'Build cancelled'
          : 'Build ended with an error'

  return (
    <div className="mx-auto max-w-3xl px-8 py-10">
      <div className="flex items-center gap-3">
        <CheckCircle2
          className={`h-7 w-7 ${status === 'done' ? 'text-vs-green' : status === 'blocked' ? 'text-vs-orange' : 'text-vs-red'}`}
        />
        <h2 className="text-2xl font-semibold text-vs-text">{headline}</h2>
      </div>

      <div className="mt-6 grid grid-cols-3 gap-3">
        <Stat label="Subtasks done" value={state.done_count} accent="text-vs-green" />
        <Stat label="Blocked" value={state.blocked_count} accent="text-vs-red" />
        <Stat label="Total time" value={`${Math.round(totalTime)}s`} accent="text-vs-text" />
      </div>

      {blocked && (
        <div className="mt-8">
          <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-vs-muted">
            blocked.md
          </h3>
          <pre className="max-h-[40vh] overflow-auto whitespace-pre-wrap rounded-md border border-vs-red/30 bg-vs-panel p-4 font-mono text-xs text-vs-text-dim">
            {blocked}
          </pre>
        </div>
      )}
    </div>
  )
}

function Stat({ label, value, accent }: { label: string; value: number | string; accent: string }) {
  return (
    <div className="rounded-md border border-vs-border bg-vs-panel p-4">
      <div className={`text-2xl font-semibold tabular-nums ${accent}`}>{value}</div>
      <div className="mt-1 text-xs text-vs-muted">{label}</div>
    </div>
  )
}
