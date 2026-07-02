import { useEffect, useRef, useState } from 'react'
import { ChevronRight } from 'lucide-react'
import { usePipelineStore } from '../../../store/pipelineStore'
import { clockTime } from '../../../lib/format'
import type { TestResultRow } from '../../../types'

export function TestResultsTab() {
  const tests = usePipelineStore((s) => s.tests)
  const bottomRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    bottomRef.current?.scrollIntoView()
  }, [tests.length])

  if (tests.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-xs text-vs-muted">
        No test runs yet.
      </div>
    )
  }
  return (
    <div className="h-full space-y-2 overflow-auto p-2">
      {tests.map((t) => (
        <Card key={t.id} test={t} />
      ))}
      <div ref={bottomRef} />
    </div>
  )
}

function Card({ test }: { test: TestResultRow }) {
  const [open, setOpen] = useState(false)
  return (
    <div
      className={`rounded border-l-2 bg-vs-panel/40 ${
        test.passed ? 'border-l-vs-green' : 'border-l-vs-red'
      }`}
    >
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs"
      >
        <ChevronRight className={`h-3 w-3 shrink-0 text-vs-muted transition-transform ${open ? 'rotate-90' : ''}`} />
        <span
          className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold ${
            test.passed ? 'bg-vs-green/15 text-vs-green' : 'bg-vs-red/15 text-vs-red'
          }`}
        >
          {test.passed ? 'PASS' : 'FAIL'}
        </span>
        <span className="truncate text-vs-text">{test.name || 'tests'}</span>
        <code className="hidden truncate text-vs-text-dim md:inline">{test.cmd}</code>
        <span className="ml-auto flex shrink-0 items-center gap-2 text-vs-muted">
          {test.exitCode !== undefined && <span>exit {test.exitCode}</span>}
          {test.duration !== undefined && <span>{test.duration.toFixed(2)}s</span>}
          <span>{clockTime(test.ts)}</span>
        </span>
      </button>
      {open && (
        <div className="px-3 pb-3">
          <code className="mb-1 block break-all font-mono text-[11px] text-vs-yellow">$ {test.cmd}</code>
          <pre className="max-h-56 overflow-auto whitespace-pre-wrap rounded bg-vs-inset p-2 font-mono text-[11px] text-vs-text-dim">
            {test.output || '(no output)'}
          </pre>
        </div>
      )}
    </div>
  )
}
