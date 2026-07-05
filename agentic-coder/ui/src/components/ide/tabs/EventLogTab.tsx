import { useEffect, useMemo, useRef, useState } from 'react'
import { usePipelineStore } from '../../../store/pipelineStore'
import { clockTime } from '../../../lib/format'
import type { LogRow } from '../../../types'

const TYPE_COLOR: Record<string, string> = {
  error: 'text-vs-red',
  'task.blocked': 'text-vs-red',
  'worker.fix_attempt': 'text-vs-red',
  'test.run': 'text-vs-yellow',
  'worker.tool_call': 'text-vs-blue',
  file_written: 'text-vs-green',
  'task.done': 'text-vs-green',
  'pipeline.complete': 'text-vs-green',
  'task.escalated': 'text-vs-orange',
  'pipeline.paused': 'text-vs-orange',
  'stage.start': 'text-vs-purple',
}

export function EventLogTab() {
  const events = usePipelineStore((s) => s.events)
  const [filter, setFilter] = useState('')
  const [autoscroll, setAutoscroll] = useState(true)
  const bottomRef = useRef<HTMLDivElement>(null)

  const types = useMemo(() => {
    const set = new Set<string>()
    events.forEach((e) => set.add(e.type))
    return Array.from(set).sort()
  }, [events])

  const filtered = filter ? events.filter((e) => e.type === filter) : events

  useEffect(() => {
    if (autoscroll) bottomRef.current?.scrollIntoView()
  }, [filtered.length, autoscroll])

  return (
    <div className="flex h-full flex-col">
      <div className="flex shrink-0 items-center gap-2 border-b border-vs-border bg-vs-panel/60 px-2 py-1">
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="rounded border border-vs-border bg-vs-inset px-1.5 py-0.5 text-xs text-vs-text outline-none"
        >
          <option value="">all types ({events.length})</option>
          {types.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
        <label className="flex items-center gap-1 text-[11px] text-vs-muted">
          <input
            type="checkbox"
            checked={autoscroll}
            onChange={(e) => setAutoscroll(e.target.checked)}
          />
          autoscroll
        </label>
      </div>

      <div className="min-h-0 flex-1 overflow-auto font-mono text-[11px]">
        {filtered.map((row) => (
          <LogLine key={row.id} row={row} />
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}

function LogLine({ row }: { row: LogRow }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="border-b border-vs-border/30">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-start gap-2 px-3 py-1 text-left hover:bg-vs-panel/40"
      >
        <span className="shrink-0 text-vs-muted">{clockTime(row.ts)}</span>
        <span className={`w-32 shrink-0 ${TYPE_COLOR[row.type] || 'text-vs-text-dim'}`}>
          [{row.type}]
        </span>
        <span className="truncate text-vs-text-dim">{row.message}</span>
      </button>
      {open && (
        <pre className="overflow-auto whitespace-pre-wrap bg-vs-inset px-3 py-2 text-[11px] text-vs-text-dim">
          {JSON.stringify({ phase: row.phase, ...row.data }, null, 2)}
        </pre>
      )}
    </div>
  )
}
