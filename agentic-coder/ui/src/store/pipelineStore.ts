import { create } from 'zustand'
import type {
  LogRow,
  ManifestEntry,
  OpenFile,
  ProjectState,
  PipelineStatus,
  SSEEvent,
  TestResultRow,
  ThinkingBlock,
  ToolCallRow,
} from '../types'
import { parseManifest } from '../lib/manifest'

const THINKING_CAP = 50
const EVENT_CAP = 1000
const TOOLCALL_CAP = 600
const TEST_CAP = 400

let _id = 0
const nextId = () => ++_id

const EMPTY_STATE: ProjectState = {
  status: 'idle',
  phase: '',
  activity: '',
  running: false,
  paused: false,
  elapsed_seconds: 0,
  subtask_elapsed_seconds: 0,
  project_dir: '',
  project_name: '',
  current_task: '',
  current_subtask: '',
  current_subtask_intent: '',
  subtask_index: 0,
  subtask_total: 0,
  task_index: 0,
  task_total: 0,
  subtask_local_index: 0,
  subtask_local_total: 0,
  done_count: 0,
  blocked_count: 0,
  pending_count: 0,
  in_progress_count: 0,
  decomposed_count: 0,
}

export interface PipelineStore {
  connected: boolean
  status: PipelineStatus
  state: ProjectState
  startedAtMs: number | null
  subtaskStartedAtMs: number | null
  startRequestedAtMs: number | null

  thinking: ThinkingBlock[]
  currentBlockId: number | null
  liveTps: number // live generation throughput (tok/s) of the active call
  genStartMs: number | null
  genChars: number
  toolCalls: ToolCallRow[]
  tests: TestResultRow[]
  testPass: number
  testFail: number
  events: LogRow[]

  manifestRaw: string
  manifest: ManifestEntry[]
  openFiles: OpenFile[]
  activeFile: string | null
  writingPath: string | null
  filesWritten: Record<string, number> // path -> last write ts (green dot + flash)

  // actions
  setConnected: (b: boolean) => void
  markStarting: () => void
  applyState: (s: ProjectState) => void
  handleEvent: (ev: SSEEvent) => void
  setManifest: (md: string) => void
  openFile: (path: string, content: string) => void
  setActiveFile: (path: string) => void
  closeFile: (path: string) => void
  endWriting: (path: string) => void
  reset: () => void
}

export const usePipelineStore = create<PipelineStore>((set, get) => ({
  connected: false,
  status: 'idle',
  state: EMPTY_STATE,
  startedAtMs: null,
  subtaskStartedAtMs: null,
  startRequestedAtMs: null,

  thinking: [],
  currentBlockId: null,
  liveTps: 0,
  genStartMs: null,
  genChars: 0,
  toolCalls: [],
  tests: [],
  testPass: 0,
  testFail: 0,
  events: [],

  manifestRaw: '',
  manifest: [],
  openFiles: [],
  activeFile: null,
  writingPath: null,
  filesWritten: {},

  setConnected: (b) => set({ connected: b }),

  markStarting: () =>
    set({
      status: 'running',
      startedAtMs: Date.now(),
      startRequestedAtMs: Date.now(),
      state: { ...EMPTY_STATE, status: 'running', running: true },
    }),

  applyState: (s) =>
    set((prev) => {
      // Don't let a stale "idle" poll undo an optimistic start.
      const justStarted =
        prev.startRequestedAtMs !== null && Date.now() - prev.startRequestedAtMs < 6000
      let status = s.status
      if (s.status === 'idle' && justStarted && (prev.status === 'running' || prev.status === 'paused')) {
        status = prev.status
      }

      const patch: Partial<PipelineStore> = { state: s, status }

      // Keep the elapsed baselines synced to the server (handles reconnects)
      // without jittering the live timer every poll.
      if ((s.running || s.status === 'paused') && s.elapsed_seconds > 0) {
        const computed = Date.now() - s.elapsed_seconds * 1000
        if (prev.startedAtMs === null || Math.abs(prev.startedAtMs - computed) > 2000) {
          patch.startedAtMs = computed
        }
      }
      if (s.subtask_elapsed_seconds > 0) {
        const computed = Date.now() - s.subtask_elapsed_seconds * 1000
        if (prev.subtaskStartedAtMs === null || Math.abs(prev.subtaskStartedAtMs - computed) > 2000) {
          patch.subtaskStartedAtMs = computed
        }
      }
      return patch
    }),

  setManifest: (md) => set({ manifestRaw: md, manifest: parseManifest(md) }),

  openFile: (path, content) =>
    set((prev) => {
      const exists = prev.openFiles.some((f) => f.path === path)
      const openFiles = exists
        ? prev.openFiles.map((f) => (f.path === path ? { path, content } : f))
        : [...prev.openFiles, { path, content }]
      return { openFiles, activeFile: path }
    }),

  setActiveFile: (path) => set({ activeFile: path }),

  endWriting: (path) =>
    set((prev) => (prev.writingPath === path ? { writingPath: null } : {})),

  closeFile: (path) =>
    set((prev) => {
      const idx = prev.openFiles.findIndex((f) => f.path === path)
      const openFiles = prev.openFiles.filter((f) => f.path !== path)
      let activeFile = prev.activeFile
      if (activeFile === path) {
        const next = openFiles[idx] || openFiles[idx - 1] || openFiles[openFiles.length - 1]
        activeFile = next ? next.path : null
      }
      return { openFiles, activeFile }
    }),

  reset: () =>
    set({
      status: 'idle',
      state: EMPTY_STATE,
      startedAtMs: null,
      subtaskStartedAtMs: null,
      startRequestedAtMs: null,
      thinking: [],
      currentBlockId: null,
      liveTps: 0,
      genStartMs: null,
      genChars: 0,
      toolCalls: [],
      tests: [],
      testPass: 0,
      testFail: 0,
      events: [],
      manifestRaw: '',
      manifest: [],
      openFiles: [],
      activeFile: null,
      writingPath: null,
      filesWritten: {},
    }),

  handleEvent: (ev) => {
    const data = ev.data || {}
    const phase = ev.phase || ''
    const ts = Date.parse(ev.timestamp) || Date.now()

    // 1) raw event log (capped, newest at the end)
    set((prev) => {
      const message = describe(ev)
      const row: LogRow = { id: nextId(), ts, type: ev.type, phase, message, data }
      const events = prev.events.length >= EVENT_CAP ? prev.events.slice(1) : prev.events.slice()
      events.push(row)
      return { events }
    })

    // 2) typed reducers
    switch (ev.type) {
      case 'stage.start':
        set((p) => ({
          startedAtMs: p.startedAtMs ?? Date.now(),
          state: { ...p.state, phase, activity: '' },
        }))
        break

      case 'llm_request': {
        const block: ThinkingBlock = {
          id: nextId(),
          phase,
          model: String(data.model || ''),
          thinking: '',
          output: '',
          complete: false,
          ts,
        }
        set((p) => {
          const thinking = [...p.thinking, block]
          if (thinking.length > THINKING_CAP) thinking.shift()
          // genStartMs is set on the FIRST token (in bumpTps), so live tok/s reflects
          // generation speed, not model load/prefill time.
          return { thinking, currentBlockId: block.id, liveTps: 0, genStartMs: null, genChars: 0 }
        })
        break
      }

      case 'llm_thinking_token': {
        const t = String(data.token || '')
        appendToCurrent(set, get, 'thinking', t)
        bumpTps(set, get, t)
        break
      }

      case 'llm_token': {
        const t = String(data.token || '')
        appendToCurrent(set, get, 'output', t)
        bumpTps(set, get, t)
        break
      }

      case 'llm_complete': {
        // Authoritative tok/s from the backend (output tokens / wall time).
        const tps = typeof data.tokens_per_second === 'number' ? data.tokens_per_second : get().liveTps
        set((p) => ({
          liveTps: tps,
          genStartMs: null,
          thinking: p.thinking.map((b) =>
            b.id === p.currentBlockId ? { ...b, complete: true, tps } : b,
          ),
        }))
        break
      }

      case 'worker.tool_call': {
        const row: ToolCallRow = {
          id: nextId(),
          ts,
          phase,
          tool: String(data.tool || ''),
          args: data.args || {},
        }
        set((p) => {
          const toolCalls =
            p.toolCalls.length >= TOOLCALL_CAP ? p.toolCalls.slice(1) : p.toolCalls.slice()
          toolCalls.push(row)
          const writingPath =
            (row.tool === 'write_file' || row.tool === 'patch_file') && row.args?.path
              ? String(row.args.path)
              : p.writingPath
          return { toolCalls, writingPath }
        })
        break
      }

      case 'worker.tool_result':
        set((p) => {
          const toolCalls = p.toolCalls.slice()
          // attach to the most recent call of the same tool still awaiting a result
          for (let i = toolCalls.length - 1; i >= 0; i--) {
            if (toolCalls[i].tool === data.tool && !toolCalls[i].result) {
              toolCalls[i] = { ...toolCalls[i], result: data as any }
              break
            }
          }
          return { toolCalls }
        })
        break

      case 'file_written': {
        const path = String(data.path || '')
        const content = typeof data.content === 'string' ? data.content : ''
        if (path) {
          set((p) => {
            const exists = p.openFiles.some((f) => f.path === path)
            const openFiles = exists
              ? p.openFiles.map((f) => (f.path === path ? { path, content } : f))
              : [...p.openFiles, { path, content }]
            return {
              openFiles,
              activeFile: path,
              writingPath: path,
              filesWritten: { ...p.filesWritten, [path]: Date.now() },
            }
          })
        }
        break
      }

      case 'test.run': {
        const passed = !!data.passed
        const row: TestResultRow = {
          id: nextId(),
          ts,
          phase,
          name: String(get().state.current_subtask || data.cmd || 'tests'),
          cmd: String(data.cmd || ''),
          passed,
          exitCode: typeof data.exit_code === 'number' ? data.exit_code : undefined,
          duration: typeof data.duration === 'number' ? data.duration : undefined,
          output: String(data.output || ''),
        }
        set((p) => {
          const tests = p.tests.length >= TEST_CAP ? p.tests.slice(1) : p.tests.slice()
          tests.push(row)
          return {
            tests,
            testPass: p.testPass + (passed ? 1 : 0),
            testFail: p.testFail + (passed ? 0 : 1),
          }
        })
        break
      }

      case 'task.start':
        set({ subtaskStartedAtMs: Date.now() })
        break

      case 'pipeline.paused':
        set((p) => ({ status: 'paused', state: { ...p.state, paused: true } }))
        break

      case 'pipeline.resumed':
        set((p) => ({ status: 'running', state: { ...p.state, paused: false } }))
        break

      case 'pipeline.complete': {
        const result = String(data.result || 'done')
        const status: PipelineStatus =
          result === 'cancelled'
            ? 'cancelled'
            : result === 'error'
              ? 'error'
              : (get().state.blocked_count || 0) > 0 && (get().state.done_count || 0) === 0
                ? 'blocked'
                : 'done'
        set((p) => ({ status, writingPath: null, state: { ...p.state, running: false } }))
        break
      }

      // A cancelled run fires pipeline.cancelled instead of pipeline.complete.
      case 'pipeline.cancelled':
        set((p) => ({ status: 'cancelled', writingPath: null, state: { ...p.state, running: false } }))
        break

      default:
        break
    }
  },
}))

// ── helpers ──────────────────────────────────────────────────────────────────
type SetFn = (
  partial:
    | Partial<PipelineStore>
    | ((s: PipelineStore) => Partial<PipelineStore>),
) => void
type GetFn = () => PipelineStore

function appendToCurrent(set: SetFn, get: GetFn, field: 'thinking' | 'output', token: string) {
  if (!token) return
  const { currentBlockId, thinking } = get()
  // If a stream arrives with no active block (e.g. after reconnect), start one.
  if (currentBlockId === null || !thinking.some((b) => b.id === currentBlockId)) {
    const block: ThinkingBlock = {
      id: nextId(),
      phase: '',
      model: '',
      thinking: field === 'thinking' ? token : '',
      output: field === 'output' ? token : '',
      complete: false,
      ts: Date.now(),
    }
    set((p) => {
      const next = [...p.thinking, block]
      if (next.length > THINKING_CAP) next.shift()
      return { thinking: next, currentBlockId: block.id }
    })
    return
  }
  set((p) => ({
    thinking: p.thinking.map((b) =>
      b.id === p.currentBlockId ? { ...b, [field]: b[field] + token } : b,
    ),
  }))
}

// Update the live tok/s estimate from streamed text (~4 chars ≈ 1 token). The
// authoritative value replaces this on llm_complete (tokens_per_second).
function bumpTps(set: SetFn, get: GetFn, token: string) {
  if (!token) return
  if (get().genStartMs === null) {
    set({ genStartMs: Date.now(), genChars: token.length })
    return
  }
  set((p) => {
    const genChars = p.genChars + token.length
    const elapsed = (Date.now() - (p.genStartMs ?? Date.now())) / 1000
    const liveTps = elapsed > 0.4 ? genChars / 4 / elapsed : p.liveTps
    return { genChars, liveTps }
  })
}

function describe(ev: SSEEvent): string {
  const d = ev.data || {}
  switch (ev.type) {
    case 'stage.start':
      return `stage: ${ev.phase} — ${d.title || ''}`
    case 'stage.end':
      return `stage complete: ${ev.phase}`
    case 'llm_request':
      return `calling ${d.model} (~${d.prompt_token_estimate ?? '?'} tok)`
    case 'llm_complete':
      return `model done (${d.total_tokens ?? '?'} tok, ${d.duration ?? '?'}s${
        d.tokens_per_second ? `, ${d.tokens_per_second} tok/s` : ''
      })`
    case 'worker.tool_call':
      return `${d.tool} ${d.args?.cmd ? '`' + d.args.cmd + '`' : d.args?.path || ''}`
    case 'worker.tool_result':
      return `${d.tool} ${'exit_code' in d ? 'exit=' + d.exit_code : d.ok ? 'ok' : 'error'}`
    case 'file_written':
      return `${d.action || 'write'} ${d.path}`
    case 'test.run':
      return `tests ${d.passed ? 'PASS' : 'FAIL'} — ${d.cmd || ''}`
    case 'task.start':
      return `subtask ${d.id} — ${d.title || ''}`
    case 'task.done':
      return `subtask ${d.id} done`
    case 'worker.fix_attempt':
      return `subtask ${d.id} attempt ${d.attempt} failed (exit ${d.exit_code})`
    case 'task.escalated':
      return `escalating ${d.id} (${d.escalations_left} left)`
    case 'task.decomposed':
      return `${d.id} decomposed into ${d.micro_count ?? '?'} micro-subtask(s)`
    case 'task.blocked':
      return `blocked ${d.id} after ${d.attempts} attempts`
    case 'manager.handoff_ready':
      return `handoff ready for ${d.subtask_id} — ${d.decision || ''}`
    case 'pipeline.complete':
      return `pipeline ${d.result}`
    case 'pipeline.cancelled':
      return 'pipeline cancelled'
    case 'pipeline.paused':
      return 'pipeline paused'
    case 'pipeline.resumed':
      return 'pipeline resumed'
    case 'error':
      return `error: ${d.message || ''}`
    case 'log':
      return String(d.message || '')
    default:
      return ev.type
  }
}
