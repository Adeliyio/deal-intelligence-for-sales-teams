import { useState } from 'react'
import { simulateScenario, ScenarioResult } from '../utils/api'

interface ScenarioSimulatorProps {
  dealId: string
}

const ACTIONS = [
  { id: 'schedule_demo', label: 'Schedule Demo', description: 'Book a product demo with the buyer' },
  { id: 'executive_outreach', label: 'Executive Outreach', description: 'Engage economic buyer or C-level' },
  { id: 'offer_discount', label: 'Offer Discount', description: 'Offer pricing concession' },
  { id: 'send_followup', label: 'Send Follow-up', description: 'Standard follow-up communication' },
  { id: 'stakeholder_mapping', label: 'Stakeholder Mapping', description: 'Multi-thread to additional contacts' },
]

export default function ScenarioSimulator({ dealId }: ScenarioSimulatorProps) {
  const [selectedAction, setSelectedAction] = useState<string>('')
  const [result, setResult] = useState<ScenarioResult | null>(null)
  const [loading, setLoading] = useState(false)

  const handleSimulate = async () => {
    if (!selectedAction) return
    setLoading(true)
    try {
      const data = await simulateScenario(dealId, selectedAction)
      setResult(data)
    } catch {
      console.error('Simulation failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-white rounded-lg border border-slate-200 p-6">
      <h3 className="font-bold text-slate-900 text-lg mb-2">Scenario Simulator</h3>
      <p className="text-sm text-slate-500 mb-4">
        Model counterfactuals: see how different actions impact win probability.
      </p>

      {/* Action selector */}
      <div className="space-y-2 mb-4">
        {ACTIONS.map((action) => (
          <label
            key={action.id}
            className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-all ${
              selectedAction === action.id
                ? 'border-primary-500 bg-primary-50'
                : 'border-slate-200 hover:border-slate-300'
            }`}
          >
            <input
              type="radio"
              name="action"
              value={action.id}
              checked={selectedAction === action.id}
              onChange={(e) => setSelectedAction(e.target.value)}
              className="mt-1"
            />
            <div>
              <p className="font-medium text-slate-900 text-sm">{action.label}</p>
              <p className="text-xs text-slate-500">{action.description}</p>
            </div>
          </label>
        ))}
      </div>

      <button
        onClick={handleSimulate}
        disabled={!selectedAction || loading}
        className="w-full py-2 px-4 bg-primary-600 text-white rounded-lg font-medium hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {loading ? 'Simulating...' : 'Run Simulation'}
      </button>

      {/* Results */}
      {result && (
        <div className="mt-6 p-4 bg-slate-50 rounded-lg">
          <h4 className="font-semibold text-sm text-slate-900 mb-3">Simulation Results</h4>

          <div className="grid grid-cols-2 gap-4 mb-3">
            <div>
              <p className="text-xs text-slate-400">Win Probability</p>
              <div className="flex items-baseline gap-2">
                <span className="text-lg font-bold text-slate-900">
                  {(result.simulated_win_probability * 100).toFixed(1)}%
                </span>
                <span
                  className={`text-sm font-medium ${
                    result.probability_delta > 0 ? 'text-emerald-600' : 'text-red-600'
                  }`}
                >
                  {result.probability_delta > 0 ? '+' : ''}
                  {(result.probability_delta * 100).toFixed(1)}%
                </span>
              </div>
            </div>
            <div>
              <p className="text-xs text-slate-400">Risk Score</p>
              <div className="flex items-baseline gap-2">
                <span className="text-lg font-bold text-slate-900">
                  {(result.simulated_risk_score * 100).toFixed(1)}%
                </span>
                <span
                  className={`text-sm font-medium ${
                    result.risk_delta < 0 ? 'text-emerald-600' : 'text-red-600'
                  }`}
                >
                  {result.risk_delta > 0 ? '+' : ''}
                  {(result.risk_delta * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>

          <p className="text-sm text-slate-600">{result.explanation}</p>
        </div>
      )}
    </div>
  )
}
