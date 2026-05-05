interface Props {
  title?: string
  backLabel?: string
  onBack?: () => void
}

export function AppHeader({ title = 'Tuberculosis Diagnostic', backLabel, onBack }: Props) {
  return (
    <header className="bg-white border-b border-gray-100 px-4 py-3 flex items-center gap-3 sticky top-0 z-10">
      {backLabel && onBack ? (
        <button onClick={onBack} className="text-primary text-sm font-medium flex items-center gap-1">
          ← {backLabel}
        </button>
      ) : (
        <>
          <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center flex-shrink-0">
            <span className="text-white text-xs font-bold">TB</span>
          </div>
          <span className="font-semibold text-gray-900 text-base">{title}</span>
        </>
      )}
    </header>
  )
}
