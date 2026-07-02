import { useEffect, useRef, useState } from 'react'
import { ChevronRight } from 'lucide-react'
import { usePipelineStore } from '../../../store/pipelineStore'
import { XTerm } from '../../common/XTerm'
import { clockTime } from '../../../lib/format'
import type { ToolCallRow } from '../../../types'

const TOOL_COLOR: Record<string, string> = {
  write_file: 'text-vs-blue',
  edit_file: 'text-vs-blue',
  read_file: 'text-vs-text-dim',
  run: 'text-vs-yellow',
}

export function ToolCallsTab() {
  const toolCalls = usePipelineStore((s) => s.toolCalls)
  const bottomRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    bottomRef.current?.scrollIntoView()
  }, [toolCalls.length])

  if (toolCalls.length === 0) {
    return <Empty text="No tool calls yet." />
  }
  return (
    <div className="h-full overflow-auto font-mono text-xs">
      {toolCalls.map((row) => (
        <Row key={row.id} row={row} />
      ))}
      <div ref={bottomRef} />
    </div>
  )
}

function Row({ row }: { row: ToolCallRow }) {
  const [open, setOpen] = useState(false)
  const res = row.result
  const isRun = row.tool === 'run'
  const exit = res?.exit_code

  return (
    <div className="border-b border-vs-border/60">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-3 py-1.5 text-left hover:bg-vs-panel/50"
      >
        <ChevronRight className={`h-3 w-3 shrink-0 text-vs-muted transition-transform ${open ? 'rotate-90' : ''}`} />
        <span className="shrink-0 text-vs-muted">{clockTime(row.ts)}</span>
        <span className={`shrink-0 font-semibold ${TOOL_COLOR[row.tool] || 'text-vs-text'}`}>
          {row.tool}
        </span>
        <span className="truncate text-vs-text-dim">
          {isRun ? row.args?.cmd : row.args?.path || ''}
        </span>
        <span className="ml-auto shrink-0">
          {res ? (
            typeof exit === 'number' ? (
              <Exit code={exit} />
            ) : res.ok ? (
              <span className="text-vs-green">✓ ok</span>
            ) : (
              <span className="text-vs-red">✗ error</span>
            )
          ) : (
            <span className="text-vs-muted">…</span>
          )}
        </span>
      </button>

      {open && (
        <div className="space-y-2 bg-vs-panel/30 px-3 pb-3 pt-1">
          {/* args */}
          <div>
            <Label>args</Label>
            <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap rounded bg-vs-inset p-2 text-[11px] text-vs-text-dim">
              {JSON.stringify(row.args, null, 2)}
            </pre>
          </div>

          {/* run output → xterm */}
          {isRun && res && (
            <div>
              <div className="mb-1 flex items-center gap-2">
                <Label>command</Label>
                <code className="rounded bg-vs-inset px-1.5 py-0.5 text-vs-yellow">{res.cmd}</code>
                {typeof exit === 'number' && <Exit code={exit} />}
              </div>
              <div className="h-44 overflow-hidden rounded border border-vs-border bg-[#0c0c0c] p-1">
                <XTerm text={termText(res)} />
              </div>
            </div>
          )}

          {/* non-run result */}
          {!isRun && res && (
            <div>
              <Label>result</Label>
              <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap rounded bg-vs-inset p-2 text-[11px] text-vs-text-dim">
                {JSON.stringify(stripBig(res), null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function termText(res: NonNullable<ToolCallRow['result']>): string {
  const parts: string[] = []
  if (res.stdout) parts.push(res.stdout)
  if (res.stderr) parts.push(res.stderr)
  if (res.error) parts.push(res.error)
  return parts.join('\n') || '(no output)'
}

function stripBig(res: Record<string, any>) {
  const { stdout, stderr, ...rest } = res
  return rest
}

function Exit({ code }: { code: number }) {
  return code === 0 ? (
    <span className="font-semibold text-vs-green">✓ {code}</span>
  ) : (
    <span className="font-semibold text-vs-red">✗ {code}</span>
  )
}

function Label({ children }: { children: React.ReactNode }) {
  return <span className="text-[10px] uppercase tracking-wider text-vs-muted">{children}</span>
}

function Empty({ text }: { text: string }) {
  return <div className="flex h-full items-center justify-center text-xs text-vs-muted">{text}</div>
}
