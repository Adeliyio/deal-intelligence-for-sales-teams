interface RiskBadgeProps {
  level: string
  className?: string
}

const riskColors: Record<string, string> = {
  low: 'bg-emerald-100 text-emerald-800',
  medium: 'bg-amber-100 text-amber-800',
  high: 'bg-red-100 text-red-800',
  critical: 'bg-red-200 text-red-900',
}

export default function RiskBadge({ level, className = '' }: RiskBadgeProps) {
  const color = riskColors[level] || 'bg-slate-100 text-slate-800'

  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${color} ${className}`}
    >
      {level.charAt(0).toUpperCase() + level.slice(1)}
    </span>
  )
}
