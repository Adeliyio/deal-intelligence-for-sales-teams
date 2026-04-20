import { useState, useEffect } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, PieChart, Pie, Cell, Legend,
} from 'recharts'
import { getEvaluationResults, getPipelineOverview, EvaluationData, PipelineOverview } from '../utils/api'

// Mock data matching real evaluation output for offline rendering
const MOCK_EVAL: EvaluationData = {
  win_model_evaluation: {
    model_name: 'Win Probability Model',
    n_test_samples: 11,
    discrimination: {
      auc_roc: 0.9643,
      auc_ci_lower: 0.7857,
      auc_ci_upper: 1.0,
      roc_curve: {
        fpr: [0, 0, 0, 0, 0, 0.143, 0.143, 1],
        tpr: [0, 0.25, 0.5, 0.75, 1, 1, 1, 1],
        thresholds: [1.95, 0.95, 0.88, 0.85, 0.78, 0.45, 0.12, 0.01],
      },
    },
    calibration: {
      brier_score_calibrated: 0.0773,
      brier_score_raw: 0.0745,
      brier_improvement: -0.0028,
      ece_calibrated: 0.1484,
      ece_raw: 0.16,
      calibration_curve: { prob_true: [0, 0.14, 1], prob_pred: [0.09, 0.12, 0.87] },
      calibration_curve_raw: { prob_true: [0, 0.14, 1], prob_pred: [0.07, 0.11, 0.92] },
    },
    threshold_analysis: {
      threshold_analysis: [
        { threshold: 0.3, precision: 0.75, recall: 1.0, f1: 0.857 },
        { threshold: 0.4, precision: 0.75, recall: 1.0, f1: 0.857 },
        { threshold: 0.45, precision: 1.0, recall: 0.75, f1: 0.857 },
        { threshold: 0.5, precision: 1.0, recall: 0.75, f1: 0.857 },
        { threshold: 0.6, precision: 1.0, recall: 0.75, f1: 0.857 },
        { threshold: 0.7, precision: 1.0, recall: 0.5, f1: 0.667 },
      ],
      optimal_threshold: 0.3,
      optimal_f1: 0.857,
    },
    shap_analysis: {
      top_features: [
        { feature: 'engagement_score', mean_shap_value: 0.312 },
        { feature: 'engagement_per_week', mean_shap_value: 0.245 },
        { feature: 'response_count', mean_shap_value: 0.198 },
        { feature: 'decay_weighted_engagement', mean_shap_value: 0.089 },
        { feature: 'silence_gap_days', mean_shap_value: 0.067 },
        { feature: 'stakeholder_count', mean_shap_value: 0.042 },
        { feature: 'deal_velocity_ratio', mean_shap_value: 0.028 },
        { feature: 'avg_response_time_hours', mean_shap_value: 0.019 },
      ],
    },
    failure_modes: {
      sparse_activity: { description: 'Deals with fewer than 5 logged activities', n_deals: 4, calibration_error: 0.0933 },
      long_cycle_enterprise: { description: 'Enterprise deals with long sales cycles', n_deals: 2, calibration_error: 0.3368 },
    },
  },
  risk_model_evaluation: {
    model_name: 'Risk Classification Model',
    discrimination: { auc_roc: 0.5 },
    calibration: { brier_score_calibrated: 0.2226, ece_calibrated: 0.0194 },
  },
  critic_ab_test: {
    n_deals: 200,
    control: { decision_quality: 0.395, false_urgency_rate: 0.0, false_urgency_count: 0, avg_calibration_error: 0.1328 },
    treatment: { decision_quality: 0.425, false_urgency_rate: 0.0, false_urgency_count: 0, avg_calibration_error: 0.1216 },
    impact: { decision_quality_delta: 0.03, false_urgency_reduction_pct: 0, calibration_improvement: 0.0112 },
    interpretation: 'The Critic Agent improved decision quality by 3.0% and risk score calibration by 0.0112 points.',
  },
}

const MOCK_OVERVIEW: PipelineOverview = {
  total_deals: 75,
  deals_at_risk: 52,
  deals_healthy: 23,
  avg_win_probability: 0.38,
  avg_risk_score: 0.67,
  risk_distribution: { low: 15, medium: 18, high: 25, critical: 17 },
}

const RISK_COLORS: Record<string, string> = {
  low: '#10b981',
  medium: '#f59e0b',
  high: '#ef4444',
  critical: '#7c2d12',
}

export default function EvalDashboard() {
  const [evalData, setEvalData] = useState<EvaluationData>(MOCK_EVAL)
  const [overview, setOverview] = useState<PipelineOverview>(MOCK_OVERVIEW)

  useEffect(() => {
    getEvaluationResults().then(setEvalData).catch(() => {})
    getPipelineOverview().then(setOverview).catch(() => {})
  }, [])

  const { win_model_evaluation: winEval, critic_ab_test: abTest } = evalData

  // ROC curve data
  const rocData = winEval.discrimination.roc_curve.fpr.map((fpr, i) => ({
    fpr: parseFloat(fpr.toFixed(3)),
    tpr: parseFloat(winEval.discrimination.roc_curve.tpr[i].toFixed(3)),
    random: parseFloat(fpr.toFixed(3)),
  }))

  // Calibration curve data
  const calData = winEval.calibration.calibration_curve.prob_pred.map((pred, i) => ({
    predicted: parseFloat(pred.toFixed(3)),
    actual: parseFloat(winEval.calibration.calibration_curve.prob_true[i].toFixed(3)),
    perfect: parseFloat(pred.toFixed(3)),
  }))

  // SHAP data
  const shapData = (winEval.shap_analysis.top_features || []).map(f => ({
    name: f.feature.replace(/_/g, ' '),
    value: parseFloat(f.mean_shap_value.toFixed(4)),
  }))

  // Risk distribution for pie chart
  const riskPieData = Object.entries(overview.risk_distribution).map(([key, value]) => ({
    name: key.charAt(0).toUpperCase() + key.slice(1),
    value,
    color: RISK_COLORS[key] || '#94a3b8',
  }))

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-900">System Evaluation</h2>
        <p className="text-sm text-slate-500 mt-1">
          Model performance metrics, calibration analysis, and Critic Agent A/B test results
        </p>
      </div>

      {/* Section 1: Key Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
        <MetricCard label="AUC-ROC" value={winEval.discrimination.auc_roc.toFixed(3)} subtitle="Win Model" good />
        <MetricCard label="Brier Score" value={winEval.calibration.brier_score_calibrated.toFixed(4)} subtitle="Calibrated" good />
        <MetricCard label="ECE" value={winEval.calibration.ece_calibrated.toFixed(4)} subtitle="Cal. Error" />
        <MetricCard label="Optimal F1" value={winEval.threshold_analysis.optimal_f1.toFixed(3)} subtitle={`@${winEval.threshold_analysis.optimal_threshold}`} good />
        <MetricCard label="Total Deals" value={overview.total_deals.toString()} subtitle="In Pipeline" />
        <MetricCard label="At Risk" value={overview.deals_at_risk.toString()} subtitle={`${((overview.deals_at_risk / overview.total_deals) * 100).toFixed(0)}% of pipeline`} bad />
      </div>

      {/* Section 2: Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ROC Curve */}
        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <h3 className="font-semibold text-slate-900 mb-1">ROC Curve</h3>
          <p className="text-xs text-slate-400 mb-4">
            AUC = {winEval.discrimination.auc_roc} [{winEval.discrimination.auc_ci_lower}, {winEval.discrimination.auc_ci_upper}]
          </p>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={rocData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="fpr" label={{ value: 'FPR', position: 'bottom', offset: -5 }} />
              <YAxis label={{ value: 'TPR', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Line type="stepAfter" dataKey="tpr" stroke="#2563eb" strokeWidth={2} name="Model" dot={false} />
              <Line type="linear" dataKey="random" stroke="#94a3b8" strokeDasharray="5 5" name="Random" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Calibration Curve */}
        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <h3 className="font-semibold text-slate-900 mb-1">Calibration Curve</h3>
          <p className="text-xs text-slate-400 mb-4">
            Predicted probability vs actual outcome frequency
          </p>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={calData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="predicted" label={{ value: 'Predicted', position: 'bottom', offset: -5 }} />
              <YAxis label={{ value: 'Actual', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Line type="monotone" dataKey="actual" stroke="#2563eb" strokeWidth={2} name="Calibrated" />
              <Line type="linear" dataKey="perfect" stroke="#94a3b8" strokeDasharray="5 5" name="Perfect" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Section 3: Critic A/B Test */}
      <div className="bg-white rounded-lg border border-slate-200 p-6">
        <h3 className="font-semibold text-slate-900 mb-1">Critic Agent A/B Test</h3>
        <p className="text-xs text-slate-400 mb-4">
          {abTest.n_deals} simulated deals — measuring impact of Critic on decision quality
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Control */}
          <div className="p-4 bg-slate-50 rounded-lg">
            <h4 className="text-sm font-medium text-slate-500 mb-3">Control (No Critic)</h4>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-slate-600">Decision Quality</span>
                <span className="font-semibold">{(abTest.control.decision_quality * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-600">Calibration Error</span>
                <span className="font-mono">{abTest.control.avg_calibration_error.toFixed(4)}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-600">False Urgency</span>
                <span>{abTest.control.false_urgency_count} cases</span>
              </div>
            </div>
          </div>

          {/* Treatment */}
          <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
            <h4 className="text-sm font-medium text-blue-700 mb-3">Treatment (With Critic)</h4>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-slate-600">Decision Quality</span>
                <span className="font-semibold text-blue-700">{(abTest.treatment.decision_quality * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-600">Calibration Error</span>
                <span className="font-mono text-blue-700">{abTest.treatment.avg_calibration_error.toFixed(4)}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-600">False Urgency</span>
                <span className="text-blue-700">{abTest.treatment.false_urgency_count} cases</span>
              </div>
            </div>
          </div>

          {/* Impact */}
          <div className="p-4 bg-emerald-50 rounded-lg border border-emerald-200">
            <h4 className="text-sm font-medium text-emerald-700 mb-3">Impact</h4>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-slate-600">Quality Delta</span>
                <span className="font-semibold text-emerald-700">+{(abTest.impact.decision_quality_delta * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-600">Calibration Gain</span>
                <span className="font-mono text-emerald-700">+{abTest.impact.calibration_improvement.toFixed(4)}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-600">False Urgency Reduction</span>
                <span className="text-emerald-700">{abTest.impact.false_urgency_reduction_pct.toFixed(0)}%</span>
              </div>
            </div>
          </div>
        </div>

        <p className="mt-4 text-sm text-slate-600 italic">{abTest.interpretation}</p>
      </div>

      {/* Section 4: SHAP Feature Importance */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <h3 className="font-semibold text-slate-900 mb-1">SHAP Feature Importance</h3>
          <p className="text-xs text-slate-400 mb-4">Mean absolute SHAP values — what drives predictions</p>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={shapData} layout="vertical" margin={{ left: 130 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="value" fill="#3b82f6" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Risk Distribution */}
        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <h3 className="font-semibold text-slate-900 mb-1">Pipeline Risk Distribution</h3>
          <p className="text-xs text-slate-400 mb-4">{overview.total_deals} deals across risk levels</p>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie
                data={riskPieData}
                cx="50%"
                cy="50%"
                outerRadius={100}
                dataKey="value"
                label={({ name, value }) => `${name}: ${value}`}
              >
                {riskPieData.map((entry, idx) => (
                  <Cell key={idx} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Section 5: Failure Modes */}
      {winEval.failure_modes && (
        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <h3 className="font-semibold text-slate-900 mb-1">Documented Failure Modes</h3>
          <p className="text-xs text-slate-400 mb-4">
            Where the model performs poorly — naming what breaks builds credibility
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(winEval.failure_modes).map(([key, mode]) => (
              <div key={key} className="p-4 bg-red-50 rounded-lg border border-red-100">
                <h4 className="font-medium text-sm text-red-900">{key.replace(/_/g, ' ')}</h4>
                <p className="text-xs text-red-700 mt-1">{mode.description}</p>
                <div className="flex gap-4 mt-2 text-xs text-red-600">
                  <span>Affected: {mode.n_deals} deals</span>
                  {mode.calibration_error !== undefined && (
                    <span>Cal Error: {mode.calibration_error.toFixed(4)}</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Section 6: Threshold Analysis */}
      <div className="bg-white rounded-lg border border-slate-200 p-6">
        <h3 className="font-semibold text-slate-900 mb-1">Threshold Analysis</h3>
        <p className="text-xs text-slate-400 mb-4">
          Performance at different operating thresholds (not just default 0.5)
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200">
                <th className="text-left py-2 px-3 text-slate-500">Threshold</th>
                <th className="text-left py-2 px-3 text-slate-500">Precision</th>
                <th className="text-left py-2 px-3 text-slate-500">Recall</th>
                <th className="text-left py-2 px-3 text-slate-500">F1</th>
              </tr>
            </thead>
            <tbody>
              {winEval.threshold_analysis.threshold_analysis.map((row) => (
                <tr
                  key={row.threshold}
                  className={`border-b border-slate-100 ${
                    row.threshold === winEval.threshold_analysis.optimal_threshold
                      ? 'bg-blue-50 font-semibold'
                      : ''
                  }`}
                >
                  <td className="py-2 px-3">{row.threshold}</td>
                  <td className="py-2 px-3">{(row.precision * 100).toFixed(1)}%</td>
                  <td className="py-2 px-3">{(row.recall * 100).toFixed(1)}%</td>
                  <td className="py-2 px-3">{(row.f1 * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function MetricCard({
  label,
  value,
  subtitle,
  good,
  bad,
}: {
  label: string
  value: string
  subtitle?: string
  good?: boolean
  bad?: boolean
}) {
  const valueColor = good ? 'text-emerald-700' : bad ? 'text-red-600' : 'text-slate-900'
  return (
    <div className="bg-white rounded-lg border border-slate-200 p-4">
      <p className="text-xs text-slate-400 uppercase tracking-wide">{label}</p>
      <p className={`text-xl font-bold mt-1 ${valueColor}`}>{value}</p>
      {subtitle && <p className="text-xs text-slate-400 mt-0.5">{subtitle}</p>}
    </div>
  )
}
