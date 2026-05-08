import { Check } from 'lucide-react'

interface Props {
  steps: number
  current: number
}

const STEP_LABELS = ['Demographics', 'Lab Results', 'Diagnosis', 'Treatment']

export function StepProgress({ steps, current }: Props) {
  return (
    <div
      className="flex items-start gap-1 px-4 py-4 bg-surface border-b border-border"
      role="progressbar"
      aria-valuenow={current}
      aria-valuemin={1}
      aria-valuemax={steps}
      aria-label={`Step ${current} of ${steps}`}
    >
      {Array.from({ length: steps }, (_, i) => {
        const step = i + 1
        const done = step < current
        const active = step === current
        return (
          <div key={step} className="flex flex-col items-center flex-1">
            <div className="flex items-center w-full">
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold font-display flex-shrink-0
                  ${done || active ? 'bg-primary text-white' : 'bg-gray-200 text-ink-muted'}`}
                aria-label={`Step ${step}${done ? ' completed' : active ? ' current' : ''}`}
              >
                {done ? <Check size={14} /> : step}
              </div>
              {i < steps - 1 && (
                <div className={`flex-1 h-0.5 mx-1 rounded ${done ? 'bg-primary' : 'bg-gray-200'}`} />
              )}
            </div>
            {STEP_LABELS[i] && (
              <p className={`text-[10px] mt-1 hidden sm:block font-medium ${active ? 'text-primary' : done ? 'text-ink-secondary' : 'text-ink-muted'}`}>
                {STEP_LABELS[i]}
              </p>
            )}
          </div>
        )
      })}
    </div>
  )
}
