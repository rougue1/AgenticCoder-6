// Shared types for the AIForge IDE frontend.

export type PipelineStatus =
  | 'idle'
  | 'running'
  | 'paused'
  | 'done'
  | 'blocked'
  | 'error'
  | 'cancelled'

// Every event type the backend emits on /events (server/events.py). Lifecycle
// events use the dotted redesign taxonomy; the model-stream layer
// (llm_request/llm_token/llm_thinking_token/llm_complete) and the tool-spec
// event (file_written) sit outside it and keep their bare names.
export const EVENT_TYPES = [
  'pipeline.start',
  'pipeline.complete',
  'pipeline.cancelled',
  'pipeline.paused',
  'pipeline.resumed',
  'stage.start',
  'stage.end',
  'manager.call_start',
  'manager.call_end',
  'manager.handoff_ready',
  'worker.call_start',
  'worker.tool_call',
  'worker.tool_result',
  'worker.fix_attempt',
  'summarizer.start',
  'summarizer.file_complete',
  'task.start',
  'task.done',
  'task.blocked',
  'task.decomposed',
  'task.escalated',
  'test.run',
  'test.passed',
  'test.failed',
  'sandbox.command_rejected',
  'sandbox.timeout',
  'preflight.check',
  'preflight.passed',
  'preflight.failed',
  'environment.setup_start',
  'environment.setup_complete',
  'llm_request',
  'llm_token',
  'llm_thinking_token',
  'llm_complete',
  'file_written',
  'error',
  'log',
] as const

export type EventType = (typeof EVENT_TYPES)[number]

export interface SSEEvent {
  type: string
  phase: string
  data: Record<string, any>
  timestamp: string
}

// Mirrors orchestrator.project_state().
export interface ProjectState {
  status: PipelineStatus
  phase: string
  activity: string
  running: boolean
  paused: boolean
  elapsed_seconds: number
  subtask_elapsed_seconds: number
  project_dir: string
  project_name: string
  current_task: string
  current_subtask: string
  current_subtask_intent: string
  subtask_index: number
  subtask_total: number
  task_index: number
  task_total: number
  subtask_local_index: number
  subtask_local_total: number
  done_count: number
  blocked_count: number
  pending_count: number
  in_progress_count: number
  decomposed_count: number
}

export interface ThinkingBlock {
  id: number
  phase: string
  model: string
  thinking: string
  output: string
  complete: boolean
  ts: number
  tps?: number // tokens/sec for this call (authoritative, set on llm_complete)
}

export interface ToolCallRow {
  id: number
  ts: number
  phase: string
  tool: string
  args: Record<string, any>
  result?: ToolResultData
}

export interface ToolResultData {
  ok: boolean
  exit_code?: number
  stdout?: string
  stderr?: string
  cmd?: string
  path?: string
  action?: string
  bytes?: number
  error?: string
  duration?: number
  [k: string]: any
}

export interface TestResultRow {
  id: number
  ts: number
  phase: string
  name: string
  cmd: string
  passed: boolean
  exitCode?: number
  duration?: number
  output: string
}

export interface LogRow {
  id: number
  ts: number
  type: string
  phase: string
  message: string
  data: Record<string, any>
}

export interface ManifestEntry {
  path: string
  desc: string
}

export interface OpenFile {
  path: string
  content: string
}
