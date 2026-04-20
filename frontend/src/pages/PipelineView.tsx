import { useState, useEffect } from 'react'
import DealCard from '../components/DealCard'
import { getPipelineOverview, getDealsList, PipelineOverview, DealListItem } from '../utils/api'

interface PipelineViewProps {
  onDealSelect: (dealId: string) => void
}

export default function PipelineView({ onDealSelect }: PipelineViewProps) {
  const [overview, setOverview] = useState<PipelineOverview | null>(null)
  const [deals, setDeals] = useState<DealListItem[]>([])
  const [filter, setFilter] = useState<string>('all')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    setError(null)

    Promise.all([getPipelineOverview(), getDealsList()])
      .then(([overviewData, dealsData]) => {
        setOverview(overviewData)
        setDeals(dealsData)
      })
      .catch((err) => {
        setError('Failed to load pipeline data. Ensure the backend is running on port 8000.')
        console.error(err)
      })
      .finally(() => setLoading(false))
  }, [])

  const filteredDeals = deals.filter((deal) => {
    if (filter === 'all') return true
    if (filter === 'at-risk') return ['high', 'critical'].includes(deal.risk_level)
    if (filter === 'healthy') return ['low', 'medium'].includes(deal.risk_level)
    return true
  })

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-slate-400">Loading pipeline data...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <p className="text-red-600 font-medium">{error}</p>
          <p className="text-sm text-slate-400 mt-2">
            Start the backend: <code className="bg-slate-100 px-2 py-1 rounded">uvicorn backend.api.main:app --reload</code>
          </p>
        </div>
      </div>
    )
  }

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
        <span className="text-sm text-slate-400 ml-auto">
          {filteredDeals.length} deals
        </span>
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
