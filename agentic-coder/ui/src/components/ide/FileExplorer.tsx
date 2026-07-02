import { useMemo, useState } from 'react'
import { ChevronDown, ChevronRight, Circle, FileCode2, Folder, FolderOpen, Pencil } from 'lucide-react'
import { usePipelineStore } from '../../store/pipelineStore'
import { buildTree, type TreeNode } from '../../lib/manifest'
import { fetchFile } from '../../api/client'

export function FileExplorer() {
  const manifest = usePipelineStore((s) => s.manifest)
  const tree = useMemo(() => buildTree(manifest), [manifest])

  return (
    <aside className="flex h-full w-full flex-col border-r border-vs-border bg-vs-panel">
      <div className="flex h-8 shrink-0 items-center px-3 text-[11px] font-semibold uppercase tracking-wider text-vs-muted">
        Explorer
      </div>
      <div className="min-h-0 flex-1 overflow-auto pb-2">
        {tree.children.length === 0 ? (
          <p className="px-3 py-2 text-xs text-vs-muted">No files yet…</p>
        ) : (
          <ul>
            {tree.children.map((child) => (
              <TreeRow key={child.path} node={child} depth={0} />
            ))}
          </ul>
        )}
      </div>
    </aside>
  )
}

function TreeRow({ node, depth }: { node: TreeNode; depth: number }) {
  const [open, setOpen] = useState(true)
  const activeFile = usePipelineStore((s) => s.activeFile)
  const writingPath = usePipelineStore((s) => s.writingPath)
  const writtenTs = usePipelineStore((s) => s.filesWritten[node.path])
  const openFile = usePipelineStore((s) => s.openFile)

  const pad = { paddingLeft: 8 + depth * 12 }

  if (node.isDir) {
    return (
      <li>
        <button
          onClick={() => setOpen((o) => !o)}
          style={pad}
          className="flex w-full items-center gap-1 py-[3px] pr-2 text-left text-[13px] text-vs-text-dim hover:bg-vs-panel2"
        >
          {open ? <ChevronDown className="h-3.5 w-3.5 shrink-0" /> : <ChevronRight className="h-3.5 w-3.5 shrink-0" />}
          {open ? (
            <FolderOpen className="h-4 w-4 shrink-0 text-vs-blue/80" />
          ) : (
            <Folder className="h-4 w-4 shrink-0 text-vs-blue/80" />
          )}
          <span className="truncate">{node.name}</span>
        </button>
        {open && (
          <ul>
            {node.children.map((child) => (
              <TreeRow key={child.path} node={child} depth={depth + 1} />
            ))}
          </ul>
        )}
      </li>
    )
  }

  const isActive = activeFile === node.path
  const isWriting = writingPath === node.path
  const wasWritten = writtenTs !== undefined

  const onClick = async () => {
    try {
      const res = await fetchFile(node.path)
      openFile(node.path, res.content || '')
    } catch {
      openFile(node.path, '')
    }
  }

  return (
    <li>
      <button
        key={`${node.path}-${writtenTs ?? 0}`}
        onClick={onClick}
        title={node.desc || node.path}
        style={pad}
        className={`flex w-full items-center gap-1.5 py-[3px] pr-2 text-left text-[13px] ${
          isActive ? 'bg-vs-accent/25 text-white' : 'text-vs-text hover:bg-vs-panel2'
        } ${wasWritten ? 'animate-flash-green' : ''}`}
      >
        <span className="w-3.5 shrink-0" />
        <FileCode2 className="h-4 w-4 shrink-0 text-vs-text-dim" />
        <span className="truncate">{node.name}</span>
        <span className="ml-auto shrink-0">
          {isWriting ? (
            <Pencil className="h-3 w-3 animate-pulse text-vs-yellow" />
          ) : wasWritten ? (
            <Circle className="h-2 w-2 fill-vs-green text-vs-green" />
          ) : null}
        </span>
      </button>
    </li>
  )
}
