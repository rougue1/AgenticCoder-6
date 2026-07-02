import { useCallback } from 'react'

// A drag handle that reports pointer deltas along one axis. The parent decides
// how to translate the delta into a new panel size (and in which direction).
export function Resizer({
  direction,
  onResize,
}: {
  direction: 'x' | 'y'
  onResize: (delta: number) => void
}) {
  const onMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault()
      const axis = direction === 'x' ? 'clientX' : 'clientY'
      let last = e[axis]
      const move = (ev: MouseEvent) => {
        const cur = ev[axis]
        onResize(cur - last)
        last = cur
      }
      const up = () => {
        window.removeEventListener('mousemove', move)
        window.removeEventListener('mouseup', up)
        document.body.style.cursor = ''
        document.body.style.userSelect = ''
      }
      window.addEventListener('mousemove', move)
      window.addEventListener('mouseup', up)
      document.body.style.cursor = direction === 'x' ? 'col-resize' : 'row-resize'
      document.body.style.userSelect = 'none'
    },
    [direction, onResize],
  )

  const cls =
    direction === 'x'
      ? 'w-[3px] cursor-col-resize hover:bg-vs-accent active:bg-vs-accent'
      : 'h-[3px] cursor-row-resize hover:bg-vs-accent active:bg-vs-accent'

  return <div onMouseDown={onMouseDown} className={`shrink-0 bg-vs-border/40 transition-colors ${cls}`} />
}
