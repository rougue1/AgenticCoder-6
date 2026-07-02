import { Ban, FilePlus2, Gauge, Pause, Play, Wifi, WifiOff } from 'lucide-react'
import { usePipelineStore } from '../../store/pipelineStore'
import { useElapsed } from '../../hooks/useElapsed'
import { phaseBadge } from '../../lib/phase'
import { hms, ms } from '../../lib/format'
import { cancelBuild, pauseBuild, resumeBuild } from '../../api/client'

export function HeaderBar() {
  const status = usePipelineStore((s) => s.status)
  const state = usePipelineStore((s) => s.state)
  const connected = usePipelineStore((s) => s.connected)
  const liveTps = usePipelineStore((s) => s.liveTps)
  const startedAtMs = usePipelineStore((s) => s.startedAtMs)
  const subtaskStartedAtMs = usePipelineStore((s) => s.subtaskStartedAtMs)
  const reset = usePipelineStore((s) => s.reset)

  const active = status === 'running' || status === 'paused'
  const elapsed = useElapsed(startedAtMs, active, state.elapsed_seconds)
  const subElapsed = useElapsed(
    subtaskStartedAtMs,
    active && state.phase === 'subtask_loop',
    state.subtask_elapsed_seconds,
  )

  const badge = phaseBadge(status, state)
  const total = state.subtask_total || 0
  const completed = (state.done_count || 0) + (state.blocked_count || 0)
  const pct = total > 0 ? Math.min(100, (completed / total) * 100) : 0

  const onStop = async () => {
    if (status === 'paused') {
      await resumeBuild().catch(() => {})
      return
    }
    if (window.confirm('Pause the build? It will finish the current operation (tool call / model stream) and then hold.')) {
      await pauseBuild().catch(() => {})
    }
  }

  const onCancel = async () => {
    if (window.confirm('This will stop the build permanently. Are you sure?')) {
      await cancelBuild().catch(() => {})
    }
  }

  const terminal = status === 'done' || status === 'blocked' || status === 'error' || status === 'cancelled'

  return (
    <header className="relative shrink-0 border-b border-vs-border bg-vs-panel">
      <div className="flex h-11 items-center gap-3 px-3">
        {/* project name */}
        <div className="flex min-w-0 items-center gap-2">
          <span className="truncate text-sm font-medium text-vs-text">
            {state.project_name || 'AIForge'}
          </span>
        </div>

        {/* phase badge */}
        <span className={`rounded px-2 py-0.5 text-[11px] font-semibold tracking-wide ${badge.cls}`}>
          {badge.label}
        </span>

        {/* subtask progress */}
        {total > 0 && (
          <span className="hidden text-xs text-vs-text-dim sm:inline">
            Task {state.task_index || 0} / {state.task_total || 0}
            <span className="mx-1.5 text-vs-muted">—</span>
            Subtask {state.subtask_local_index || 0} / {state.subtask_local_total || 0}
          </span>
        )}

        <div className="ml-auto flex items-center gap-4">
          {/* timers + throughput */}
          <div className="flex items-center gap-3 font-mono text-xs">
            {liveTps > 0 && (
              <div className="flex items-center gap-1.5" title="Model generation throughput (tokens/sec)">
                <Gauge className="h-3.5 w-3.5 text-vs-purple" />
                <span className="tabular-nums text-vs-text">{liveTps.toFixed(1)}</span>
                <span className="text-vs-muted">tok/s</span>
              </div>
            )}
            <div className="flex items-center gap-1.5" title="Total elapsed">
              <span className="text-vs-muted">total</span>
              <span className="tabular-nums text-vs-text">{hms(elapsed)}</span>
            </div>
            {state.phase === 'subtask_loop' && (
              <div className="flex items-center gap-1.5" title="Current subtask">
                <span className="text-vs-muted">subtask</span>
                <span className="tabular-nums text-vs-blue">{ms(subElapsed)}</span>
              </div>
            )}
          </div>

          {/* connection */}
          <span
            title={connected ? 'connected' : 'disconnected'}
            className={connected ? 'text-vs-green' : 'text-vs-red'}
          >
            {connected ? <Wifi className="h-4 w-4" /> : <WifiOff className="h-4 w-4" />}
          </span>

          {/* controls */}
          {!terminal && (
            <>
              <button
                onClick={onStop}
                className="inline-flex items-center gap-1.5 rounded border border-vs-border px-2.5 py-1 text-xs text-vs-text hover:border-vs-accent hover:text-white"
              >
                {status === 'paused' ? (
                  <>
                    <Play className="h-3.5 w-3.5" /> Resume
                  </>
                ) : (
                  <>
                    <Pause className="h-3.5 w-3.5" /> Stop
                  </>
                )}
              </button>
              <button
                onClick={onCancel}
                className="inline-flex items-center gap-1.5 rounded border border-vs-red/40 px-2.5 py-1 text-xs text-vs-red hover:bg-vs-red/10"
              >
                <Ban className="h-3.5 w-3.5" /> Cancel
              </button>
            </>
          )}

          {(status === 'done' || status === 'blocked' || status === 'error' || status === 'cancelled') && (
            <button
              onClick={reset}
              className="inline-flex items-center gap-1.5 rounded bg-vs-accent px-2.5 py-1 text-xs font-medium text-white hover:bg-vs-accent-hover"
            >
              <FilePlus2 className="h-3.5 w-3.5" /> New Project
            </button>
          )}
        </div>
      </div>

      {/* thin progress bar beneath the header */}
      <div className="h-0.5 w-full bg-vs-border/50">
        <div
          className="h-full bg-vs-accent transition-[width] duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
    </header>
  )
}
