import { useState, useEffect } from 'react'
import DealCard from '../components/DealCard'
import { getPipelineOverview, PipelineOverview } from '../utils/api'

interface PipelineViewProps {
  onDealSelect: (dealId: string) => void
}

// Mock deal data for display (in production, fetched from API)
const MOCK_DEALS = [
  { deal_id: 'DEAL-0001', win_probability: 0.09, risk_level: 'critical', deal_value: 392800, stage: 'closed_lost', silence_gap_days: 10, engagement_per_week: 0.168 },
  { deal_id: 'DEAL-0005', win_probability: 0.86, risk_level: 'low', deal_value: 325300, stage: 'closed_won', silence_gap_days: 2, engagement_per_week: 1.711 },
  { deal_id: 'DEAL-0010', win_probability: 0.45, risk_level: 'medium', deal_value: 18900, stage: 'qualification', silence_gap_days: 8, engagement_per_week: 0.35 },
  { deal_id: 'DEAL-0015', win_probability: 0.22, risk_level: 'high', deal_value: 67500, stage: 'proposal', silence_gap_days: 18, engagement_per_week: 0.12 },
  { deal_id: 'DEAL-0020', win_probability: 0.71, risk_level: 'low', deal_value: 12400, stage: 'negotiation', silence_gap_days: 3, engagement_per_week: 0.89 },
  { deal_id: 'DEAL-0025', win_probability: 0.33, risk_level: 'high', deal_value: 156000, stage: 'qualification', silence_gap_days: 22, engagement_per_week: 0.08 },
  { deal_id: 'DEAL-0030', win_probability: 0.58, risk_level: 'medium', deal_value: 43200, stage: 'proposal', silence_gap_days: 6, engagement_per_week: 0.45 },
  { deal_id: 'DEAL-0035', win_probability: 0.12, risk_level: 'critical', deal_value: 89000, stage: 'negotiation', silence_gap_days: 31, engagement_per_week: 0.05 },
]

export default function PipelineView({ onDealSelect }: PipelineViewProps) {
  const [overview, setOverview] = useState<PipelineOverview | null>(null)
  const [filter, setFilter] = useState<string>('all')

  useEffect(() => {
    getPipelineOverview()
      .then(setOverview)
      .catch(() => {
        // Use mock data if API unavailable
        setOverview({
          total_deals: 75,
          deals_at_risk: 52,
          deals_healthy: 23,
          avg_win_probability: 0.38,
          avg_risk_score: 0.67,
          risk_distribution: { low: 15, medium: 18, high: 25, critical: 17 },
        })
      })
  }, [])

  const filteredDeals = MOCK_DEALS.filter((deal) => {
    if (filter === 'all') return true
    if (filter === 'at-risk') return ['high', 'critical'].includes(deal.risk_level)
    if (filter === 'healthy') return ['low', 'medium'].includes(deal.risk_level)
    return true
  })

  return (
    <div>
      {/* Pipeline metrics */}
      {overview && (
        <div className="grid grid-cols-4 gap-4 mb-6">
          <MetricCard label="Total Deals" value={overview.total_deals} />
          <MetricCard
            label="At Risk"
            value={overview.deals_at_risk}
            color="text-red-600"
          />
          <MetricCard
            label="Avg Win Probability"
            value={`${(overview.avg_win_probability * 100).toFixed(0)}%`}
          />
          <MetricCard
            label="Avg Risk Score"
            value={`${(overview.avg_risk_score * 100).toFixed(0)}%`}
            color="text-amber-600"
          />
        </div>
      )}

      {/* Filters */}
      <div className="flex items-center gap-3 mb-6">
        <h2 className="text-lg font-bold text-slate-900">Pipeline</h2>
        <div className="flex gap-2 ml-4">
          {['all', 'at-risk', 'healthy'].map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-1 text-sm rounded-full transition-colors ${
                filter === f
                  ? 'bg-primary-600 text-white'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              {f === 'all' ? 'All' : f === 'at-risk' ? 'At Risk' : 'Healthy'}
            </button>
          ))}
        </div>
      </div>

      {/* Deal grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {filteredDeals.map((deal) => (
          <DealCard
            key={deal.deal_id}
            dealId={deal.deal_id}
            winProbability={deal.win_probability}
            riskLevel={deal.risk_level}
            dealValue={deal.deal_value}
            stage={deal.stage}
            silenceGapDays={deal.silence_gap_days}
            engagementPerWeek={deal.engagement_per_week}
            onClick={() => onDealSelect(deal.deal_id)}
          />
        ))}
      </div>
    </div>
  )
}

function MetricCard({
  label,
  value,
  color = 'text-slate-900',
}: {
  label: string
  value: string | number
  color?: string
}) {
  return (
    <div className="bg-white rounded-lg border border-slate-200 p-4">
      <p className="text-xs text-slate-400 uppercase tracking-wide">{label}</p>
      <p className={`text-2xl font-bold mt-1 ${color}`}>{value}</p>
    </div>
  )
}
