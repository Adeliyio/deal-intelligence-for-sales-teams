import { useState, useEffect } from 'react'
import { analyzeDeal, DealAnalysis } from '../utils/api'
import WinProbabilityChart from '../components/WinProbabilityChart'
import AgentDebateView from '../components/AgentDebateView'
import ScenarioSimulator from '../components/ScenarioSimulator'
import RiskBadge from '../components/RiskBadge'

interface DealDetailProps {
  dealId: string
  onBack: () => void
}

export default function DealDetail({ dealId, onBack }: DealDetailProps) {
  const [analysis, setAnalysis] = useState<DealAnalysis | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'overview' | 'agents' | 'simulator'>('overview')

  useEffect(() => {
    setLoading(true)
    setError(null)
    analyzeDeal(dealId)
      .then(setAnalysis)
      .catch((err) => {
        setError('Failed to analyze deal. Ensure the backend is running.')
        console.error(err)
      })
      .finally(() => setLoading(false))
  }, [dealId])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-pulse text-slate-400">Analyzing deal...</div>
      </div>
    )
  }

  if (error || !analysis) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <p className="text-red-600 font-medium">{error || 'Failed to load deal analysis'}</p>
          <button onClick={onBack} className="mt-4 text-primary-600 hover:text-primary-800 text-sm font-medium">
            &larr; Back to Pipeline
          </button>
        </div>
      </div>
    )
  }

  const { predictions } = analysis

  return (
    <div>
      {/* Header */}
      <div className="flex items-center gap-4 mb-6">
        <button
          onClick={onBack}
          className="text-sm text-primary-600 hover:text-primary-800 font-medium"
        >
          &larr; Back to Pipeline
        </button>
        <h2 className="text-xl font-bold text-slate-900">{dealId}</h2>
        <RiskBadge level={predictions.risk_level} />
        {predictions.low_confidence_warning && (
          <span className="text-xs bg-amber-100 text-amber-700 px-2 py-1 rounded-full">
            Low Confidence
          </span>
        )}
      </div>

      {/* Tabs */}
      <div className="border-b border-slate-200 mb-6">
        <div className="flex gap-6">
          {[
            { id: 'overview' as const, label: 'Overview & Predictions' },
            { id: 'agents' as const, label: 'Agent Debate' },
            { id: 'simulator' as const, label: 'Scenario Simulator' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`pb-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === tab.id
                  ? 'border-primary-600 text-primary-600'
                  : 'border-transparent text-slate-500 hover:text-slate-700'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left: ML predictions */}
          <div className="lg:col-span-2">
            <WinProbabilityChart
              winProbability={predictions.win_probability}
              confidenceLower={predictions.confidence_lower}
              confidenceUpper={predictions.confidence_upper}
              featureImportance={analysis.feature_importance}
            />

            {/* Strategy recommendations */}
            {analysis.strategy && (
              <div className="mt-6 bg-white rounded-lg border border-slate-200 p-6">
                <h4 className="text-sm font-medium text-slate-500 mb-3">
                  Strategy Recommendations
                </h4>
                <div className="space-y-2">
                  {analysis.strategy.split('\n').map((line, i) => (
                    <p
                      key={i}
                      className={`text-sm ${
                        line.startsWith('URGENT')
                          ? 'text-red-700 font-semibold'
                          : 'text-slate-700'
                      }`}
                    >
                      {line}
                    </p>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Right: Historical matches */}
          <div>
            <div className="bg-white rounded-lg border border-slate-200 p-6">
              <h4 className="text-sm font-medium text-slate-500 mb-4">
                Similar Historical Deals
              </h4>
              {analysis.historical_matches.length > 0 ? (
                <div className="space-y-3">
                  {analysis.historical_matches.map((match) => (
                    <div
                      key={match.deal_id}
                      className="p-3 bg-slate-50 rounded-lg"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium text-sm text-slate-900">
                          {match.deal_id}
                        </span>
                        <span
                          className={`text-xs font-medium ${
                            match.outcome === 'won'
                              ? 'text-emerald-600'
                              : 'text-red-600'
                          }`}
                        >
                          {match.outcome}
                        </span>
                      </div>
                      <div className="text-xs text-slate-500">
                        <span>Similarity: {(match.similarity * 100).toFixed(0)}%</span>
                        {match.silence_gap_days != null && (
                          <span className="ml-3">
                            Silence: {match.silence_gap_days.toFixed(0)}d
                          </span>
                        )}
                        {match.duration_days != null && (
                          <span className="ml-3">
                            Duration: {match.duration_days}d
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-slate-400">
                  No comparable historical deals found.
                </p>
              )}
            </div>

            {/* Risk summary */}
            <div className="mt-4 bg-white rounded-lg border border-slate-200 p-6">
              <h4 className="text-sm font-medium text-slate-500 mb-3">Risk Summary</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-600">Risk Score</span>
                  <span className="font-semibold">
                    {(predictions.risk_score * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-600">At Risk</span>
                  <span className={predictions.is_at_risk ? 'text-red-600 font-semibold' : 'text-emerald-600'}>
                    {predictions.is_at_risk ? 'Yes' : 'No'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-600">Confidence Width</span>
                  <span className="font-mono text-xs">
                    &plusmn;{(predictions.confidence_width * 100 / 2).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'agents' && (
        <AgentDebateView
          leadAssessment={analysis.lead_assessment}
          riskAnalysis={analysis.risk_analysis}
          strategy={analysis.strategy}
          communication={analysis.communication}
          criticReview={analysis.critic_review}
        />
      )}

      {activeTab === 'simulator' && <ScenarioSimulator dealId={dealId} />}
    </div>
  )
}
