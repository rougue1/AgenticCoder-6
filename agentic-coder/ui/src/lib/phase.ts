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

const PLANNING_LABELS: Record<string, string> = {
  intake: 'INTAKE',
  requirements: 'REQUIREMENTS',
  stack_decider: 'STACK',
  architect: 'ARCHITECT',
  sdd_generator: 'SDD',
  task_planner: 'PLANNING',
  reviewer: 'REVIEW',
  setup: 'SETUP',
  resume: 'RESUME',
}

const ACTIVITY: Record<string, Badge> = {
  plan: { label: 'PLAN', cls: COLORS.blue },
  implement: { label: 'IMPLEMENT', cls: COLORS.green },
  test: { label: 'TEST', cls: COLORS.yellow },
  fix: { label: 'FIX', cls: COLORS.orange },
  escalate: { label: 'ESCALATE', cls: COLORS.orange },
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
  if (phase === 'reviewer') return { label: 'REVIEW', cls: COLORS.purple }
  const label = PLANNING_LABELS[phase] || (phase ? phase.toUpperCase() : 'STARTING')
  return { label, cls: COLORS.blue }
}
