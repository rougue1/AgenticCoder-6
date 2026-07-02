import { useState } from 'react'
import { PanelBottomClose, PanelRightClose, Terminal as TerminalIcon } from 'lucide-react'
import { Resizer } from '../common/Resizer'
import { HeaderBar } from './HeaderBar'
import { FileExplorer } from './FileExplorer'
import { MainPanel } from './MainPanel'
import { ThinkingPanel } from './ThinkingPanel'
import { BottomPanel } from './BottomPanel'

const clamp = (n: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, n))

// View B — the full VSCode-style IDE. Four regions: header, explorer, main +
// thinking, and a full-width bottom panel. Explorer/thinking/bottom are all
// drag-resizable; thinking and bottom are collapsible.
export function IDEView() {
  const [explorerW, setExplorerW] = useState(220)
  const [thinkW, setThinkW] = useState(300)
  const [thinkOpen, setThinkOpen] = useState(true)
  const [bottomH, setBottomH] = useState(220)
  const [bottomOpen, setBottomOpen] = useState(true)

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-vs-bg text-vs-text">
      <HeaderBar />

      <div className="flex min-h-0 flex-1 flex-col">
        {/* upper region: explorer | main | thinking */}
        <div className="flex min-h-0 flex-1">
          <div style={{ width: explorerW }} className="shrink-0">
            <FileExplorer />
          </div>
          <Resizer direction="x" onResize={(d) => setExplorerW((w) => clamp(w + d, 150, 480))} />

          <div className="min-w-0 flex-1">
            <MainPanel />
          </div>

          {thinkOpen ? (
            <>
              <Resizer direction="x" onResize={(d) => setThinkW((w) => clamp(w - d, 200, 620))} />
              <div style={{ width: thinkW }} className="shrink-0">
                <ThinkingPanel onCollapse={() => setThinkOpen(false)} />
              </div>
            </>
          ) : (
            <button
              onClick={() => setThinkOpen(true)}
              title="Show Model Thinking"
              className="flex w-9 shrink-0 flex-col items-center gap-2 border-l border-vs-border bg-vs-panel py-3 text-vs-muted hover:text-vs-text"
            >
              <PanelRightClose className="h-4 w-4" />
              <span className="[writing-mode:vertical-rl] text-[11px] tracking-wide">Thinking</span>
            </button>
          )}
        </div>

        {/* bottom region (full width) */}
        {bottomOpen ? (
          <>
            <Resizer
              direction="y"
              onResize={(d) => setBottomH((h) => clamp(h - d, 120, window.innerHeight * 0.7))}
            />
            <div style={{ height: bottomH }} className="shrink-0">
              <BottomPanel onCollapse={() => setBottomOpen(false)} />
            </div>
          </>
        ) : (
          <button
            onClick={() => setBottomOpen(true)}
            title="Show bottom panel"
            className="flex h-8 shrink-0 items-center gap-2 border-t border-vs-border bg-vs-panel px-3 text-xs text-vs-muted hover:text-vs-text"
          >
            <TerminalIcon className="h-3.5 w-3.5" />
            Tool Calls · Test Results · Event Log
            <PanelBottomClose className="ml-1 h-3.5 w-3.5" />
          </button>
        )}
      </div>
    </div>
  )
}
