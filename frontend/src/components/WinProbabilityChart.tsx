import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ErrorBar,
} from 'recharts'
import { FeatureImportance } from '../utils/api'

interface WinProbabilityChartProps {
  winProbability: number
  confidenceLower: number
  confidenceUpper: number
  featureImportance: FeatureImportance[]
}

export default function WinProbabilityChart({
  winProbability,
  confidenceLower,
  confidenceUpper,
  featureImportance,
}: WinProbabilityChartProps) {
  const chartData = featureImportance.slice(0, 8).map((f) => ({
    name: f.feature.replace(/_/g, ' '),
    importance: parseFloat(f.importance_pct.toFixed(1)),
  }))

  return (
    <div className="space-y-6">
      {/* Win probability gauge */}
      <div className="bg-white rounded-lg border border-slate-200 p-6">
        <h4 className="text-sm font-medium text-slate-500 mb-2">Win Probability</h4>
        <div className="flex items-baseline gap-3">
          <span className="text-4xl font-bold text-slate-900">
            {(winProbability * 100).toFixed(0)}%
          </span>
          <span className="text-sm text-slate-400">
            CI: [{(confidenceLower * 100).toFixed(0)}% - {(confidenceUpper * 100).toFixed(0)}%]
          </span>
        </div>

        {/* Confidence interval visualization */}
        <div className="mt-4 relative">
          <div className="w-full bg-slate-100 rounded-full h-4">
            {/* CI range */}
            <div
              className="absolute h-4 bg-blue-100 rounded-full"
              style={{
                left: `${confidenceLower * 100}%`,
                width: `${(confidenceUpper - confidenceLower) * 100}%`,
              }}
            />
            {/* Point estimate */}
            <div
              className="absolute h-4 w-1 bg-primary-600 rounded"
              style={{ left: `${winProbability * 100}%` }}
            />
          </div>
          <div className="flex justify-between mt-1 text-xs text-slate-400">
            <span>0%</span>
            <span>50%</span>
            <span>100%</span>
          </div>
        </div>
      </div>

      {/* Feature importance chart */}
      {chartData.length > 0 && (
        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <h4 className="text-sm font-medium text-slate-500 mb-4">
            Feature Importance (What Drives This Prediction)
          </h4>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={chartData} layout="vertical" margin={{ left: 120 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" unit="%" />
              <YAxis type="category" dataKey="name" width={110} tick={{ fontSize: 12 }} />
              <Tooltip formatter={(value: number) => `${value}%`} />
              <Bar dataKey="importance" fill="#3b82f6" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}
