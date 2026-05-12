interface Props {
  probability: number
  size?: 'sm' | 'md' | 'lg'
}

export function riskLabel(p: number): 'HIGH' | 'MED' | 'LOW' {
  if (p >= 0.6) return 'HIGH'
  if (p >= 0.35) return 'MED'
  return 'LOW'
}

export function riskColor(p: number) {
  const label = riskLabel(p)
  if (label === 'HIGH') return { bg: 'bg-risk-high/10', text: 'text-risk-high', border: 'border-risk-high/30' }
  if (label === 'MED')  return { bg: 'bg-risk-med/10',  text: 'text-risk-med',  border: 'border-risk-med/30'  }
  return                       { bg: 'bg-risk-low/10',  text: 'text-risk-low',  border: 'border-risk-low/30'  }
}

export function RiskBadge({ probability, size = 'md' }: Props) {
  const label = riskLabel(probability)
  const { bg, text } = riskColor(probability)
  const sizeClass = size === 'sm'
    ? 'text-xs px-1.5 py-0.5'
    : size === 'lg'
    ? 'text-sm px-3 py-1.5'
    : 'text-xs px-2 py-1'
  const humanLabel = label === 'HIGH' ? 'High risk' : label === 'MED' ? 'Medium risk' : 'Low risk'
  return (
    <span
      className={`${bg} ${text} font-semibold rounded-full tracking-wide ${sizeClass}`}
      aria-label={`${humanLabel}: ${Math.round(probability * 100)}%`}
    >
      {label}
    </span>
  )
}
