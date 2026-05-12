import { useEffect, useCallback, useRef } from 'react'
import { X, ChevronLeft, ChevronRight } from 'lucide-react'
import { useXrayUrl } from '../lib/useXrayUrl'

interface Props {
  ids: string[]
  index: number
  onClose: () => void
  onPrev: () => void
  onNext: () => void
}

export function XrayLightbox({ ids, index, onClose, onPrev, onNext }: Props) {
  const url = useXrayUrl(ids[index])
  const hasPrev = index > 0
  const hasNext = index < ids.length - 1

  const dialogRef = useRef<HTMLDivElement>(null)
  const closeButtonRef = useRef<HTMLButtonElement>(null)

  // Move focus into dialog on open; return focus to the activating element on close
  useEffect(() => {
    const trigger = document.activeElement as HTMLElement | null
    closeButtonRef.current?.focus()
    return () => { trigger?.focus() }
  }, [])

  // Mark background siblings as `inert` so virtual cursors (NVDA+Chrome, etc.) cannot escape
  useEffect(() => {
    const dialog = dialogRef.current
    if (!dialog) return
    const inerted: Element[] = []
    let el: Element | null = dialog.parentElement?.firstElementChild ?? null
    while (el) {
      if (el !== dialog) {
        el.setAttribute('inert', '')
        inerted.push(el)
      }
      el = el.nextElementSibling
    }
    return () => { inerted.forEach(s => s.removeAttribute('inert')) }
  }, [])

  // Tab-trap: keep keyboard focus cycling within the dialog
  useEffect(() => {
    function handleTab(e: KeyboardEvent) {
      if (e.key !== 'Tab' || !dialogRef.current) return
      const focusable = Array.from(
        dialogRef.current.querySelectorAll<HTMLElement>(
          'button:not([disabled]), [href], input:not([disabled]), [tabindex]:not([tabindex="-1"])',
        ),
      )
      if (focusable.length === 0) return
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault()
        last.focus()
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault()
        first.focus()
      }
    }
    document.addEventListener('keydown', handleTab)
    return () => document.removeEventListener('keydown', handleTab)
  }, [])

  const handleKey = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape') { onClose() }
      else if (e.key === 'ArrowLeft' && hasPrev) { e.preventDefault(); onPrev() }
      else if (e.key === 'ArrowRight' && hasNext) { e.preventDefault(); onNext() }
    },
    [onClose, onPrev, onNext, hasPrev, hasNext],
  )

  useEffect(() => {
    document.addEventListener('keydown', handleKey)
    return () => document.removeEventListener('keydown', handleKey)
  }, [handleKey])

  // Prevent body scroll while open; restore on close
  useEffect(() => {
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => { document.body.style.overflow = prev }
  }, [])

  return (
    <div
      ref={dialogRef}
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink-base/80 backdrop-blur-sm print:hidden"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label={`X-ray image viewer, ${index + 1} of ${ids.length}`}
    >
      {/* Visually-hidden live region announces position and load state to screen readers */}
      <span
        className="sr-only"
        aria-live="polite"
        aria-atomic="true"
      >
        {url
          ? `X-ray ${index + 1} of ${ids.length} loaded`
          : `Loading X-ray ${index + 1} of ${ids.length}`}
      </span>

      {/* Visual image counter (hidden from AT — dialog aria-label carries the count) */}
      <p
        className="absolute top-4 left-1/2 -translate-x-1/2 text-white/70 text-sm tabular-nums pointer-events-none"
        aria-hidden="true"
      >
        {index + 1} / {ids.length}
      </p>

      {/* Close — p-3 for ≥44px touch target; bg-white/30 for 3:1 non-text contrast */}
      <button
        ref={closeButtonRef}
        type="button"
        onClick={onClose}
        aria-label="Close image viewer"
        className="absolute top-4 right-4 z-10 text-white bg-white/30 hover:bg-white/40 rounded-full p-3 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white"
      >
        <X size={20} aria-hidden="true" />
      </button>

      {/* Previous */}
      {hasPrev && (
        <button
          type="button"
          onClick={e => { e.stopPropagation(); onPrev() }}
          aria-label="Previous image"
          className="absolute left-4 z-10 text-white bg-white/30 hover:bg-white/40 rounded-full p-3 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white"
        >
          <ChevronLeft size={24} aria-hidden="true" />
        </button>
      )}

      {/* Image area — click-inside stops propagation so clicking image doesn't close */}
      <div
        className="max-h-[90vh] max-w-[90vw] flex items-center justify-center"
        onClick={e => e.stopPropagation()}
      >
        {url ? (
          <img
            src={url}
            alt={`X-ray ${index + 1} of ${ids.length}`}
            className="max-h-[90vh] max-w-[90vw] object-contain rounded-lg shadow-2xl"
          />
        ) : (
          <div className="w-64 h-64 bg-white/10 rounded-lg animate-pulse flex items-center justify-center">
            <p className="text-white/50 text-sm" aria-hidden="true">Loading…</p>
          </div>
        )}
      </div>

      {/* Next */}
      {hasNext && (
        <button
          type="button"
          onClick={e => { e.stopPropagation(); onNext() }}
          aria-label="Next image"
          className="absolute right-4 z-10 text-white bg-white/30 hover:bg-white/40 rounded-full p-3 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white"
        >
          <ChevronRight size={24} aria-hidden="true" />
        </button>
      )}
    </div>
  )
}
