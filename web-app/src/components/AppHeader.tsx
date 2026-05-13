import { ChevronLeft } from 'lucide-react'

interface Props {
  title?: string
  backLabel?: string
  onBack?: () => void
}

export function AppHeader({ title = 'Tuberculosis Diagnostic System', backLabel, onBack }: Props) {
  return (
    <header className="lg:hidden bg-surface border-b border-border px-4 py-3 flex items-center gap-3 sticky top-0 z-10">
      {backLabel && onBack ? (
        <button
          onClick={onBack}
          className="text-primary text-sm font-medium flex items-center gap-1 min-h-[44px] px-1"
          aria-label={`Back to ${backLabel}`}
        >
          <ChevronLeft size={16} />
          {backLabel}
        </button>
      ) : (
        <>
          <img
            src="/logo.png"
            alt="TB Logo"
            className="w-9 h-9 rounded-xl object-contain flex-shrink-0"
          />
          <span className="font-semibold text-ink-base text-base font-display">{title}</span>
        </>
      )}
    </header>
  )
}