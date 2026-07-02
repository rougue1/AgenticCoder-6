// Thin fetch wrappers around the FastAPI backend. All paths are same-origin
// relative (the backend serves this bundle and, in dev, Vite proxies them), so
// no base URL is needed. Override with VITE_API_BASE if hosting elsewhere.
import type { ProjectState } from '../types'

const BASE = (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, '') || ''

export const apiUrl = (path: string) => `${BASE}${path}`

async function postJson(path: string, body?: unknown): Promise<any> {
  const res = await fetch(apiUrl(path), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  let data: any = {}
  try {
    data = await res.json()
  } catch {
    /* empty body */
  }
  if (!res.ok && data?.ok !== false) {
    throw new Error(data?.error || `${path} failed (${res.status})`)
  }
  return data
}

export function startBuild(prompt: string, projectDir?: string) {
  return postJson('/start', { prompt, project_dir: projectDir || null })
}

export function resumeBuild(projectDir?: string) {
  return postJson('/resume', { project_dir: projectDir || null })
}

export function pauseBuild() {
  return postJson('/pause')
}

export function cancelBuild() {
  return postJson('/cancel')
}

export async function fetchState(signal?: AbortSignal): Promise<ProjectState> {
  const res = await fetch(apiUrl('/project/state'), { signal })
  if (!res.ok) throw new Error(`/project/state failed (${res.status})`)
  return res.json()
}

export async function fetchManifest(signal?: AbortSignal): Promise<string> {
  const res = await fetch(apiUrl('/project/manifest'), { signal })
  if (!res.ok) throw new Error(`/project/manifest failed (${res.status})`)
  return res.text()
}

export async function fetchFile(
  path: string,
  signal?: AbortSignal,
): Promise<{ content: string; exists: boolean }> {
  const res = await fetch(apiUrl(`/file?path=${encodeURIComponent(path)}`), { signal })
  if (!res.ok && res.status !== 400) throw new Error(`/file failed (${res.status})`)
  return res.json()
}
