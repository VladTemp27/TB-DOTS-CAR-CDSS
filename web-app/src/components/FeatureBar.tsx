interface Props {
  feature: string
  delta: number
  direction: 'risk' | 'protective'
  maxDelta: number
}

export function FeatureBar({ feature, delta, direction, maxDelta }: Props) {
  const pct = maxDelta > 0 ? (delta / maxDelta) * 100 : 0
  const isRisk = direction === 'risk'

  return (
    <div className="flex items-center gap-3 py-1">
      <span className="text-sm text-gray-700 w-36 text-right flex-shrink-0">{feature}</span>
      <div className="flex-1 flex items-center gap-1">
        <div className="flex-1 h-4 bg-gray-100 rounded overflow-hidden">
          <div
            className={`h-full rounded ${isRisk ? 'bg-red-400' : 'bg-green-400'}`}
            style={{ width: `${Math.min(pct, 100)}%` }}
          />
        </div>
        <span className={`text-xs font-semibold w-10 ${isRisk ? 'text-red-500' : 'text-green-500'}`}>
          {isRisk ? '+' : '-'}{(delta * 100).toFixed(0)}%
        </span>
      </div>
    </div>
  )
}
