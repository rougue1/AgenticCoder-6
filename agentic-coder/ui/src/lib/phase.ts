import type { PipelineStatus, ProjectState } from '../types'

export interface Badge {
  label: string
  cls: string // tailwind classes for the pill
}

const COLORS: Record<string, string> = {
  blue: 'bg-vs-blue/15 text-vs-blue ring-1 ring-vs-blue/40',
  green: 'bg-vs-green/15 text-vs-green ring-1 ring-vs-green/40',
  yellow: 'bg-vs-yellow/15 text-vs-yellow ring-1 ring-vs-yellow/40',
  orange: 'bg-vs-orange/20 text-vs-orange ring-1 ring-vs-orange/40',
  red: 'bg-vs-red/15 text-vs-red ring-1 ring-vs-red/40',
  purple: 'bg-vs-purple/15 text-vs-purple ring-1 ring-vs-purple/40',
  muted: 'bg-vs-muted/15 text-vs-text-dim ring-1 ring-vs-muted/30',
}

// Mirrors orchestrator/states.py's PIPELINE_ORDER phase labels (the two-tier
// Manager/Worker redesign — resolution -> preflight -> stack -> environment ->
// requirements -> architecture -> task_planner -> subtask_loop -> final_review).
const PLANNING_LABELS: Record<string, string> = {
  resolution: 'RESOLVING MODELS',
  preflight: 'PREFLIGHT',
  stack: 'STACK',
  environment: 'ENVIRONMENT',
  requirements: 'REQUIREMENTS',
  architecture: 'ARCHITECTURE',
  task_planner: 'PLANNING',
}

// Mirrors services.Progress.activity during the subtask loop (Manager handoff
// -> Worker implement/test/fix -> escalate/decompose -> Analyst summarize).
const ACTIVITY: Record<string, Badge> = {
  handoff: { label: 'HANDOFF', cls: COLORS.blue },
  implement: { label: 'IMPLEMENT', cls: COLORS.green },
  test: { label: 'TEST', cls: COLORS.yellow },
  fix: { label: 'FIX', cls: COLORS.orange },
  escalate: { label: 'ESCALATE', cls: COLORS.orange },
  decompose: { label: 'DECOMPOSE', cls: COLORS.purple },
  summarize: { label: 'SUMMARIZE', cls: COLORS.blue },
}

// Color-coded phase badge: planning=blue, coding=green, testing=yellow,
// escalation=orange, blocked=red (spec §Header Bar).
export function phaseBadge(status: PipelineStatus, state: ProjectState): Badge {
  if (status === 'blocked') return { label: 'BLOCKED', cls: COLORS.red }
  if (status === 'error') return { label: 'ERROR', cls: COLORS.red }
  if (status === 'cancelled') return { label: 'CANCELLED', cls: COLORS.muted }
  if (status === 'done') return { label: 'DONE', cls: COLORS.green }
  if (status === 'paused') return { label: 'PAUSED', cls: COLORS.muted }

  const phase = state.phase || ''
  if (phase === 'subtask_loop') {
    return ACTIVITY[state.activity] || { label: 'IMPLEMENT', cls: COLORS.green }
  }
  if (phase === 'final_review') return { label: 'REVIEW', cls: COLORS.purple }
  const label = PLANNING_LABELS[phase] || (phase ? phase.toUpperCase() : 'STARTING')
  return { label, cls: COLORS.blue }
}
