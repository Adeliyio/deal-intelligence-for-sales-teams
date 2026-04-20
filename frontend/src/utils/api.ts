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

export default api
