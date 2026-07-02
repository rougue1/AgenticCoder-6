import { useEffect, useRef } from 'react'
import { usePipelineStore } from '../store/pipelineStore'
import { apiUrl, fetchManifest, fetchState } from '../api/client'
import { EVENT_TYPES, type SSEEvent } from '../types'

const POLL_MS = 5000
const MAX_BACKOFF = 30000

// Opens the /events SSE stream, funnels every event into the Zustand store, and
// keeps the connection alive across backend restarts with exponential backoff.
// Also polls /project/state as a reconnection-recovery fallback and refreshes
// the file manifest whenever a file is written.
export function useEventStream() {
  const setConnected = usePipelineStore((s) => s.setConnected)
  const handleEvent = usePipelineStore((s) => s.handleEvent)
  const applyState = usePipelineStore((s) => s.applyState)
  const setManifest = usePipelineStore((s) => s.setManifest)

  const esRef = useRef<EventSource | null>(null)
  const attemptsRef = useRef(0)
  const closedRef = useRef(false)
  const reconnectTimer = useRef<number | null>(null)
  const manifestTimer = useRef<number | null>(null)
  const stateTimer = useRef<number | null>(null)

  useEffect(() => {
    closedRef.current = false

    const refreshState = () => {
      if (stateTimer.current) window.clearTimeout(stateTimer.current)
      stateTimer.current = window.setTimeout(() => {
        fetchState().then(applyState).catch(() => {})
      }, 120)
    }
    const refreshManifest = () => {
      if (manifestTimer.current) window.clearTimeout(manifestTimer.current)
      manifestTimer.current = window.setTimeout(() => {
        fetchManifest().then(setManifest).catch(() => {})
      }, 150)
    }

    const dispatch = (raw: string) => {
      let ev: SSEEvent
      try {
        ev = JSON.parse(raw)
      } catch {
        return
      }
      handleEvent(ev)
      // Side effects that need a fresh server snapshot / manifest.
      if (ev.type === 'file_written') refreshManifest()
      // Refresh the rich snapshot at the transitions that change the phase
      // badge / progress (activity isn't carried on the SSE stream itself).
      if (
        ev.type === 'subtask_start' ||
        ev.type === 'subtask_done' ||
        ev.type === 'subtask_failed' ||
        ev.type === 'blocked' ||
        ev.type === 'escalation' ||
        ev.type === 'test_run' ||
        ev.type === 'stage_start' ||
        ev.type === 'pipeline_complete'
      ) {
        refreshState()
      }
    }

    const connect = () => {
      if (closedRef.current) return
      const es = new EventSource(apiUrl('/events'))
      esRef.current = es

      es.onopen = () => {
        attemptsRef.current = 0
        setConnected(true)
        // Pull authoritative state + manifest right after (re)connecting.
        fetchState().then(applyState).catch(() => {})
        fetchManifest().then(setManifest).catch(() => {})
      }

      // The backend tags every event with `event: <type>`, so the default
      // `message` handler never fires — attach one listener per known type.
      for (const type of EVENT_TYPES) {
        es.addEventListener(type, (e) => {
          const data = (e as MessageEvent).data
          if (typeof data === 'string') dispatch(data)
        })
      }

      es.onerror = () => {
        // Native connection error (distinct from a server-sent `error` event,
        // which carries `.data` and is handled by the listener above).
        setConnected(false)
        es.close()
        if (esRef.current === es) esRef.current = null
        if (closedRef.current) return
        const backoff = Math.min(MAX_BACKOFF, 1000 * 2 ** attemptsRef.current)
        attemptsRef.current += 1
        reconnectTimer.current = window.setTimeout(connect, backoff)
      }
    }

    connect()

    // Polling fallback (covers missed events after a flaky reconnect).
    const poll = window.setInterval(() => {
      fetchState().then(applyState).catch(() => {})
    }, POLL_MS)

    return () => {
      closedRef.current = true
      window.clearInterval(poll)
      if (reconnectTimer.current) window.clearTimeout(reconnectTimer.current)
      if (manifestTimer.current) window.clearTimeout(manifestTimer.current)
      if (stateTimer.current) window.clearTimeout(stateTimer.current)
      esRef.current?.close()
      esRef.current = null
    }
  }, [setConnected, handleEvent, applyState, setManifest])
}
