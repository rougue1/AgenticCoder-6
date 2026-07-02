import { useEventStream } from './hooks/useEventStream'
import { usePipelineStore } from './store/pipelineStore'
import { LaunchScreen } from './components/LaunchScreen'
import { IDEView } from './components/ide/IDEView'

export default function App() {
  // Single SSE connection + state polling for the whole app.
  useEventStream()
  const status = usePipelineStore((s) => s.status)

  // View A (launch) only while idle; View B (IDE) for every active/terminal state.
  return status === 'idle' ? <LaunchScreen /> : <IDEView />
}
