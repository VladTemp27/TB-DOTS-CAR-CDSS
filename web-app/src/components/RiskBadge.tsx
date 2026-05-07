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
  if (label === 'HIGH') return { bg: 'bg-red-100', text: 'text-red-600', border: 'border-red-200' }
  if (label === 'MED') return { bg: 'bg-orange-100', text: 'text-orange-600', border: 'border-orange-200' }
  return { bg: 'bg-green-100', text: 'text-green-600', border: 'border-green-200' }
}

export function RiskBadge({ probability, size = 'md' }: Props) {
  const label = riskLabel(probability)
  const { bg, text } = riskColor(probability)
  const sizeClass = size === 'sm' ? 'text-xs px-1.5 py-0.5' : size === 'lg' ? 'text-sm px-3 py-1.5' : 'text-xs px-2 py-1'
  return (
    <span className={`${bg} ${text} font-bold rounded ${sizeClass}`}>{label}</span>
  )
}
