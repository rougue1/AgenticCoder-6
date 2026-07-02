import { useMemo } from 'react'
import hljs from 'highlight.js'
import { langForPath } from '../../lib/format'

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}

// Read-only, syntax-highlighted file view with a line-number gutter. When
// `showCursor` is set a blinking write-cursor is appended (used while a file is
// streaming in from `file_written`).
export function CodeView({
  path,
  content,
  showCursor = false,
}: {
  path: string
  content: string
  showCursor?: boolean
}) {
  const lang = langForPath(path)

  const html = useMemo(() => {
    let out: string
    try {
      out =
        lang !== 'plaintext' && hljs.getLanguage(lang)
          ? hljs.highlight(content, { language: lang }).value
          : escapeHtml(content)
    } catch {
      out = escapeHtml(content)
    }
    if (showCursor) out += '<span class="aiforge-cursor"></span>'
    return out
  }, [content, lang, showCursor])

  const lineCount = Math.max(1, content.split('\n').length)
  const gutter = useMemo(
    () => Array.from({ length: lineCount }, (_, i) => i + 1).join('\n'),
    [lineCount],
  )

  return (
    <div className="flex min-h-full font-mono text-[12.5px] leading-[1.5]">
      <pre
        className="select-none whitespace-pre py-3 pl-4 pr-3 text-right text-vs-muted/70 tabular-nums"
        aria-hidden
      >
        {gutter}
      </pre>
      <pre className="hljs flex-1 overflow-x-auto whitespace-pre py-3 pr-6 pl-2 text-vs-text">
        <code dangerouslySetInnerHTML={{ __html: html }} />
      </pre>
    </div>
  )
}
