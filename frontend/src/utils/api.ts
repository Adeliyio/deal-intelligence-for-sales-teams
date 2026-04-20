import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
})

export interface MLPrediction {
  win_probability: number
  confidence_lower: number
  confidence_upper: number
  confidence_width: number
  risk_score: number
  risk_level: string
  is_at_risk: boolean
  low_confidence_warning: boolean
}

export interface HistoricalMatch {
  deal_id: string
  outcome: string
  similarity: number
  engagement_per_week?: number
  silence_gap_days?: number
  duration_days?: number
}

export interface FeatureImportance {
  feature: string
  importance: number
  importance_pct: number
}

export interface DealAnalysis {
  deal_id: string
  predictions: MLPrediction
  lead_assessment?: string
  risk_analysis?: string
  strategy?: string
  communication?: string
  critic_review?: string
  historical_matches: HistoricalMatch[]
  feature_importance: FeatureImportance[]
}

export interface PipelineOverview {
  total_deals: number
  deals_at_risk: number
  deals_healthy: number
  avg_win_probability: number
  avg_risk_score: number
  risk_distribution: Record<string, number>
}

export interface ScenarioResult {
  deal_id: string
  action: string
  current_win_probability: number
  simulated_win_probability: number
  probability_delta: number
  current_risk_score: number
  simulated_risk_score: number
  risk_delta: number
  explanation: string
}

export const analyzeDeal = async (dealId: string): Promise<DealAnalysis> => {
  const response = await api.post('/analyze-deal', { deal_id: dealId })
  return response.data
}

export const predictOutcome = async (dealId: string) => {
  const response = await api.post('/predict-outcome', { deal_id: dealId })
  return response.data
}

export const generateStrategy = async (dealId: string) => {
  const response = await api.post('/generate-strategy', { deal_id: dealId })
  return response.data
}

export const simulateScenario = async (
  dealId: string,
  action: string,
  parameters?: Record<string, unknown>
): Promise<ScenarioResult> => {
  const response = await api.post('/simulate-scenario', {
    deal_id: dealId,
    action,
    parameters,
  })
  return response.data
}

export const getPipelineOverview = async (): Promise<PipelineOverview> => {
  const response = await api.get('/pipeline-overview')
  return response.data
}

export interface DealListItem {
  deal_id: string
  win_probability: number
  risk_level: string
  risk_score: number
  deal_value: number
  stage: string
  silence_gap_days: number
  engagement_per_week: number
  industry: string
  deal_size_bucket: string
  outcome: string
}

export interface EvaluationData {
  win_model_evaluation: {
    model_name: string
    n_test_samples: number
    discrimination: {
      auc_roc: number
      auc_ci_lower: number
      auc_ci_upper: number
      roc_curve: { fpr: number[]; tpr: number[]; thresholds: number[] }
    }
    calibration: {
      brier_score_calibrated: number
      brier_score_raw: number
      brier_improvement: number
      ece_calibrated: number
      ece_raw: number
      calibration_curve: { prob_true: number[]; prob_pred: number[] }
      calibration_curve_raw: { prob_true: number[]; prob_pred: number[] }
    }
    threshold_analysis: {
      threshold_analysis: Array<{
        threshold: number
        precision: number
        recall: number
        f1: number
      }>
      optimal_threshold: number
      optimal_f1: number
    }
    shap_analysis: {
      top_features?: Array<{ feature: string; mean_shap_value: number }>
    }
    failure_modes?: Record<string, { description: string; n_deals: number; calibration_error?: number }>
  }
  risk_model_evaluation: {
    model_name: string
    discrimination: { auc_roc: number }
    calibration: { brier_score_calibrated: number; ece_calibrated: number }
  }
  critic_ab_test: {
    n_deals: number
    control: {
      decision_quality: number
      false_urgency_rate: number
      false_urgency_count: number
      avg_calibration_error: number
    }
    treatment: {
      decision_quality: number
      false_urgency_rate: number
      false_urgency_count: number
      avg_calibration_error: number
    }
    impact: {
      decision_quality_delta: number
      false_urgency_reduction_pct: number
      calibration_improvement: number
    }
    interpretation: string
  }
}

export const getEvaluationResults = async (): Promise<EvaluationData> => {
  const response = await api.get('/evaluation-results')
  return response.data
}

export const getDealsList = async (): Promise<DealListItem[]> => {
  const response = await api.get('/deals-list')
  return response.data.deals
}

export default api
