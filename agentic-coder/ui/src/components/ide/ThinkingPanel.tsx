import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { Brain, ChevronRight, PanelRightClose, Pause, Play } from 'lucide-react'
import { usePipelineStore } from '../../store/pipelineStore'
import type { ThinkingBlock } from '../../types'

export function ThinkingPanel({ onCollapse }: { onCollapse: () => void }) {
  const thinking = usePipelineStore((s) => s.thinking)
  const currentId = usePipelineStore((s) => s.currentBlockId)
  const liveTps = usePipelineStore((s) => s.liveTps)
  const [autoscroll, setAutoscroll] = useState(true)
  const scrollRef = useRef<HTMLDivElement>(null)
  // Whether the user is currently parked at the bottom. We only follow new tokens
  // when they are — so scrolling up (to read history or collapse a block) is no
  // longer fought by a scroll-to-bottom on every streamed token.
  const atBottomRef = useRef(true)

  const onScroll = () => {
    const el = scrollRef.current
    if (!el) return
    atBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48
  }

  // Sticky auto-scroll: follow the newest tokens ONLY when autoscroll is on AND the
  // user hasn't scrolled away from the bottom. The Pause button is a hard override.
  const lastLen = thinking[thinking.length - 1]
  const tick = (lastLen?.thinking.length || 0) + (lastLen?.output.length || 0)
  useLayoutEffect(() => {
    if (autoscroll && atBottomRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [tick, autoscroll, thinking.length])

  return (
    <aside className="flex h-full w-full flex-col border-l border-vs-border bg-vs-inset">
      <div className="flex h-8 shrink-0 items-center gap-2 border-b border-vs-border px-2">
        <Brain className="h-4 w-4 text-vs-purple" />
        <span className="text-[11px] font-semibold uppercase tracking-wider text-vs-muted">
          Model Thinking
        </span>
        <div className="ml-auto flex items-center gap-1">
          {liveTps > 0 && (
            <span className="mr-1 font-mono text-[10px] tabular-nums text-vs-purple" title="tokens/sec">
              {liveTps.toFixed(1)} tok/s
            </span>
          )}
          <button
            onClick={() => {
              // Turning autoscroll back on should snap to the bottom and resume following.
              if (!autoscroll) atBottomRef.current = true
              setAutoscroll((v) => !v)
            }}
            title={autoscroll ? 'Pause scroll' : 'Resume scroll'}
            className="rounded p-1 text-vs-muted hover:bg-vs-panel2 hover:text-vs-text"
          >
            {autoscroll ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
          </button>
          <button
            onClick={onCollapse}
            title="Collapse"
            className="rounded p-1 text-vs-muted hover:bg-vs-panel2 hover:text-vs-text"
          >
            <PanelRightClose className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      <div ref={scrollRef} onScroll={onScroll} className="min-h-0 flex-1 overflow-auto px-2 py-2">
        {thinking.length === 0 ? (
          <p className="px-1 py-2 text-xs text-vs-muted">No model output yet…</p>
        ) : (
          thinking.map((b) => (
            <Block
              key={b.id}
              block={b}
              isCurrent={b.id === currentId}
              liveTps={b.id === currentId ? liveTps : undefined}
            />
          ))
        )}
      </div>
    </aside>
  )
}

function Block({
  block,
  isCurrent,
  liveTps,
}: {
  block: ThinkingBlock
  isCurrent: boolean
  liveTps?: number
}) {
  // The active block is always expanded; history items collapse, click to expand.
  const [open, setOpen] = useState(isCurrent)
  useEffect(() => {
    if (isCurrent) setOpen(true)
  }, [isCurrent])

  const preview = (block.output || block.thinking || '').slice(0, 60).replace(/\n/g, ' ')
  // tok/s: live while generating, the authoritative value once complete.
  const tps = block.complete ? block.tps : liveTps

  return (
    <div className={`mb-2 rounded border ${isCurrent ? 'border-vs-purple/40' : 'border-vs-border'} bg-vs-panel/40`}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-1.5 px-2 py-1 text-left"
      >
        <ChevronRight className={`h-3 w-3 shrink-0 text-vs-muted transition-transform ${open ? 'rotate-90' : ''}`} />
        <span className="text-[10px] uppercase tracking-wide text-vs-blue">{block.phase || 'llm'}</span>
        {block.model && <span className="truncate text-[10px] text-vs-muted">· {block.model}</span>}
        {isCurrent && !block.complete && (
          <span className="ml-1 h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-vs-purple" />
        )}
        {!open && preview && (
          <span className="ml-1 min-w-0 flex-1 truncate text-[11px] text-vs-muted">{preview}…</span>
        )}
        {typeof tps === 'number' && tps > 0 && (
          <span className="ml-auto shrink-0 pl-2 font-mono text-[10px] tabular-nums text-vs-purple/80">
            {tps.toFixed(1)} tok/s
          </span>
        )}
      </button>

      {open && (
        <div className="px-2.5 pb-2.5">
          {block.thinking && (
            <p className="whitespace-pre-wrap break-words font-mono text-[11.5px] italic leading-relaxed text-vs-muted">
              {block.thinking}
            </p>
          )}
          {block.thinking && block.output && (
            <div className="my-2 flex items-center gap-2">
              <div className="h-px flex-1 bg-vs-border" />
              <span className="text-[9px] uppercase tracking-wider text-vs-muted">output</span>
              <div className="h-px flex-1 bg-vs-border" />
            </div>
          )}
          {block.output && (
            <p className="whitespace-pre-wrap break-words font-mono text-[11.5px] leading-relaxed text-vs-text">
              {block.output}
            </p>
          )}
        </div>
      )}
    </div>
  )
}
