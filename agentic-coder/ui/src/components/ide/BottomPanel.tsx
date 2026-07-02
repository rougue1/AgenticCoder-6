import { useState } from 'react'
import { ChevronDown } from 'lucide-react'
import { usePipelineStore } from '../../store/pipelineStore'
import { ToolCallsTab } from './tabs/ToolCallsTab'
import { TestResultsTab } from './tabs/TestResultsTab'
import { EventLogTab } from './tabs/EventLogTab'

type Tab = 'tools' | 'tests' | 'log'

export function BottomPanel({ onCollapse }: { onCollapse: () => void }) {
  const [tab, setTab] = useState<Tab>('tools')
  const toolCount = usePipelineStore((s) => s.toolCalls.length)
  const testPass = usePipelineStore((s) => s.testPass)
  const testFail = usePipelineStore((s) => s.testFail)
  const eventCount = usePipelineStore((s) => s.events.length)

  return (
    <section className="flex h-full flex-col border-t border-vs-border bg-vs-bg">
      <div className="flex h-8 shrink-0 items-stretch border-b border-vs-border bg-vs-panel">
        <TabButton active={tab === 'tools'} onClick={() => setTab('tools')}>
          Tool Calls {toolCount > 0 && <Count>{toolCount}</Count>}
        </TabButton>
        <TabButton active={tab === 'tests'} onClick={() => setTab('tests')}>
          Test Results{' '}
          {testPass + testFail > 0 && (
            <span className="ml-1 text-[11px]">
              (<span className="text-vs-green">{testPass}✓</span>{' '}
              <span className="text-vs-red">{testFail}✗</span>)
            </span>
          )}
        </TabButton>
        <TabButton active={tab === 'log'} onClick={() => setTab('log')}>
          Event Log {eventCount > 0 && <Count>{eventCount}</Count>}
        </TabButton>

        <button
          onClick={onCollapse}
          title="Collapse panel"
          className="ml-auto px-3 text-vs-muted hover:bg-vs-panel2 hover:text-vs-text"
        >
          <ChevronDown className="h-4 w-4" />
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-hidden">
        {tab === 'tools' && <ToolCallsTab />}
        {tab === 'tests' && <TestResultsTab />}
        {tab === 'log' && <EventLogTab />}
      </div>
    </section>
  )
}

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean
  onClick: () => void
  children: React.ReactNode
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center border-r border-vs-border px-3 text-xs ${
        active
          ? 'border-b-2 border-b-vs-accent bg-vs-bg text-vs-text'
          : 'text-vs-text-dim hover:text-vs-text'
      }`}
    >
      {children}
    </button>
  )
}

function Count({ children }: { children: React.ReactNode }) {
  return <span className="ml-1 rounded bg-vs-panel2 px-1.5 text-[10px] text-vs-muted">{children}</span>
}
