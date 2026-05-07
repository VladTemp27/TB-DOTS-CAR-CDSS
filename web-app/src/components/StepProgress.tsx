interface Props {
  steps: number
  current: number
}

export function StepProgress({ steps, current }: Props) {
  return (
    <div className="flex items-center gap-1 px-4 py-3">
      {Array.from({ length: steps }, (_, i) => {
        const step = i + 1
        const done = step < current
        const active = step === current
        return (
          <div key={step} className="flex items-center flex-1">
            <div
              className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-semibold flex-shrink-0
                ${done ? 'bg-primary text-white' : active ? 'bg-primary text-white' : 'bg-gray-200 text-gray-500'}`}
            >
              {done ? '✓' : step}
            </div>
            {i < steps - 1 && (
              <div className={`flex-1 h-0.5 mx-1 ${done ? 'bg-primary' : 'bg-gray-200'}`} />
            )}
          </div>
        )
      })}
    </div>
  )
}
