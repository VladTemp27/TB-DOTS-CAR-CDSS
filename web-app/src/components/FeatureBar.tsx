interface Props {
  feature: string
  delta: number
  direction: 'risk' | 'protective'
  maxDelta: number
  value?: string
}

export function FeatureBar({ feature, delta, direction, maxDelta, value }: Props) {
  const pct = maxDelta > 0 ? (delta / maxDelta) * 100 : 0
  const isRisk = direction === 'risk'

  return (
    <div className="flex items-center gap-3 py-1.5">
      <div className="w-36 md:w-48 text-right flex-shrink-0 leading-tight">
        <div className="text-sm text-ink-secondary">{feature}</div>
        {value && <div className="text-xs text-ink-muted">{value}</div>}
      </div>
      <div className="flex-1 flex items-center gap-2">
        <div className="flex-1 h-3 bg-gray-100 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full ${isRisk ? 'bg-risk-high/70' : 'bg-risk-low/70'}`}
            style={{ width: `${Math.min(pct, 100)}%` }}
          />
        </div>
        <span className={`text-xs font-semibold tabular-nums w-12 text-right ${isRisk ? 'text-risk-high' : 'text-risk-low'}`}>
          {isRisk ? '+' : '-'}{(delta * 100).toFixed(0)}%
        </span>
      </div>
    </div>
  )
}
