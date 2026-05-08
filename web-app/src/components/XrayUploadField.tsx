import { useRef, useState, useEffect } from 'react'
import { ImagePlus, X } from 'lucide-react'

const ALLOWED_MIME = ['image/jpeg', 'image/png', 'image/webp']

interface FileItem {
  file: File
  url: string
}

interface Props {
  onChange: (files: File[]) => void
  max?: number
}

export function XrayUploadField({ onChange, max = 10 }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [items, setItems] = useState<FileItem[]>([])
  const [dragging, setDragging] = useState(false)

  // itemsRef mirrors items and is eagerly updated in event handlers so that
  // rapid consecutive calls (drops before re-render) and the unmount cleanup
  // both see the latest snapshot without depending on state flush timing.
  const itemsRef = useRef<FileItem[]>([])
  itemsRef.current = items

  useEffect(() => {
    return () => { itemsRef.current.forEach(i => URL.revokeObjectURL(i.url)) }
  }, [])

  function mergeFiles(incoming: File[]) {
    const valid = incoming.filter(f => ALLOWED_MIME.includes(f.type))
    if (valid.length === 0) return
    const newItems = valid.map(f => ({ file: f, url: URL.createObjectURL(f) }))
    const combined = [...itemsRef.current, ...newItems]
    combined.slice(max).forEach(i => URL.revokeObjectURL(i.url))
    const next = combined.slice(0, max)
    itemsRef.current = next
    setItems(next)
    onChange(next.map(i => i.file))
  }

  function removeFile(index: number) {
    URL.revokeObjectURL(itemsRef.current[index].url)
    const next = itemsRef.current.filter((_, i) => i !== index)
    itemsRef.current = next
    setItems(next)
    onChange(next.map(i => i.file))
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    setDragging(false)
    mergeFiles(Array.from(e.dataTransfer.files))
  }

  return (
    <div className="space-y-3">
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload chest X-ray images — drag and drop or click to browse"
        onClick={() => inputRef.current?.click()}
        onKeyDown={e => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            inputRef.current?.click()
          }
        }}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        className={`flex flex-col items-center justify-center gap-2 px-4 py-8 border-2 border-dashed rounded-2xl cursor-pointer select-none transition-colors
          ${dragging
            ? 'border-primary bg-primary-50'
            : 'border-border bg-surface hover:border-primary/50 hover:bg-primary-50/50'
          }
          focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-1`}
      >
        <ImagePlus
          size={28}
          className={dragging ? 'text-primary' : 'text-ink-muted'}
          aria-hidden="true"
        />
        <p className="text-sm font-medium text-ink-secondary">
          Drag & drop X-rays or click to browse
        </p>
        <p className="text-xs text-ink-muted">JPEG, PNG, WebP · up to {max} images</p>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        multiple
        className="hidden"
        aria-hidden="true"
        onChange={e => { mergeFiles(Array.from(e.target.files ?? [])); e.target.value = '' }}
      />

      {items.length > 0 && (
        <div
          className="grid grid-cols-3 sm:grid-cols-4 gap-2"
          role="list"
          aria-label={`${items.length} X-ray image${items.length === 1 ? '' : 's'} selected`}
        >
          {items.map((item, i) => (
            <div
              key={item.url}
              role="listitem"
              className="relative aspect-square rounded-xl overflow-hidden bg-gray-100"
            >
              <img
                src={item.url}
                alt={item.file.name}
                className="w-full h-full object-cover"
              />
              <button
                type="button"
                onClick={e => { e.stopPropagation(); removeFile(i) }}
                aria-label={`Remove ${item.file.name}`}
                className="absolute top-1 right-1 bg-ink-base/70 hover:bg-ink-base text-white rounded-full p-0.5 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white"
              >
                <X size={12} aria-hidden="true" />
              </button>
              <p
                className="absolute bottom-0 left-0 right-0 bg-ink-base/60 text-white text-[9px] px-1 py-0.5 truncate"
                aria-hidden="true"
              >
                {item.file.name}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
