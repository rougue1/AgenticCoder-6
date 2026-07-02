import type { ManifestEntry } from '../types'

// Parse `.agent/file_manifest.md` (produced by context/manifest.py). Each file
// line looks like:  - `src/app.py` — FastAPI entrypoint
const LINE_RE = /^\s*-\s*`([^`]+)`\s*(?:[—-]\s*(.*))?$/

export function parseManifest(md: string): ManifestEntry[] {
  const out: ManifestEntry[] = []
  for (const raw of md.split('\n')) {
    const m = raw.match(LINE_RE)
    if (m) out.push({ path: m[1].trim(), desc: (m[2] || '').trim() })
  }
  out.sort((a, b) => a.path.localeCompare(b.path))
  return out
}

// ── Tree model for the explorer ──────────────────────────────────────────────
export interface TreeNode {
  name: string
  path: string // full path for files; dir path for folders
  isDir: boolean
  desc?: string
  children: TreeNode[]
}

export function buildTree(entries: ManifestEntry[]): TreeNode {
  const root: TreeNode = { name: '', path: '', isDir: true, children: [] }
  for (const entry of entries) {
    const parts = entry.path.split('/').filter(Boolean)
    let node = root
    let acc = ''
    parts.forEach((part, i) => {
      acc = acc ? `${acc}/${part}` : part
      const isLeaf = i === parts.length - 1
      let child = node.children.find((c) => c.name === part && c.isDir === !isLeaf)
      if (!child) {
        child = {
          name: part,
          path: acc,
          isDir: !isLeaf,
          desc: isLeaf ? entry.desc : undefined,
          children: [],
        }
        node.children.push(child)
      }
      node = child
    })
  }
  sortTree(root)
  return root
}

function sortTree(node: TreeNode) {
  node.children.sort((a, b) => {
    if (a.isDir !== b.isDir) return a.isDir ? -1 : 1 // folders first
    return a.name.localeCompare(b.name)
  })
  node.children.forEach(sortTree)
}
