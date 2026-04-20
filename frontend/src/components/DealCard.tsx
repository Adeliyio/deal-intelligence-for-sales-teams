import RiskBadge from './RiskBadge'

interface DealCardProps {
  dealId: string
  winProbability: number
  riskLevel: string
  dealValue: number
  stage: string
  silenceGapDays: number
  engagementPerWeek: number
  onClick: () => void
}

export default function DealCard({
  dealId,
  winProbability,
  riskLevel,
  dealValue,
  stage,
  silenceGapDays,
  engagementPerWeek,
  onClick,
}: DealCardProps) {
  const probColor =
    winProbability > 0.6
      ? 'text-emerald-600'
      : winProbability > 0.35
        ? 'text-amber-600'
        : 'text-red-600'

  return (
    <div
      onClick={onClick}
      className="bg-white rounded-lg border border-slate-200 p-4 hover:shadow-md hover:border-primary-300 cursor-pointer transition-all"
    >
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="font-semibold text-slate-900">{dealId}</h3>
          <p className="text-sm text-slate-500 capitalize">{stage.replace('_', ' ')}</p>
        </div>
        <RiskBadge level={riskLevel} />
      </div>

      <div className="grid grid-cols-2 gap-3 text-sm">
        <div>
          <p className="text-slate-400 text-xs">Win Probability</p>
          <p className={`font-bold text-lg ${probColor}`}>
            {(winProbability * 100).toFixed(0)}%
          </p>
        </div>
        <div>
          <p className="text-slate-400 text-xs">Deal Value</p>
          <p className="font-semibold text-slate-900">
            ${dealValue.toLocaleString()}
          </p>
        </div>
        <div>
          <p className="text-slate-400 text-xs">Silence Gap</p>
          <p className="text-slate-700">{silenceGapDays.toFixed(0)}d</p>
        </div>
        <div>
          <p className="text-slate-400 text-xs">Engagement/wk</p>
          <p className="text-slate-700">{engagementPerWeek.toFixed(2)}</p>
        </div>
      </div>

      {/* Win probability bar */}
      <div className="mt-3">
        <div className="w-full bg-slate-100 rounded-full h-2">
          <div
            className={`h-2 rounded-full ${
              winProbability > 0.6
                ? 'bg-emerald-500'
                : winProbability > 0.35
                  ? 'bg-amber-500'
                  : 'bg-red-500'
            }`}
            style={{ width: `${winProbability * 100}%` }}
          />
        </div>
      </div>
    </div>
  )
}
