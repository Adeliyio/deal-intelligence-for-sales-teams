"""
Pydantic models for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


# --- Request Models ---

class DealAnalysisRequest(BaseModel):
    """Request body for /analyze-deal endpoint."""
    deal_id: str = Field(..., description="Unique deal identifier")


class PredictOutcomeRequest(BaseModel):
    """Request body for /predict-outcome endpoint."""
    deal_id: Optional[str] = Field(None, description="Deal ID to look up features")
    features: Optional[Dict[str, Any]] = Field(
        None, description="Deal features dict (if not using deal_id lookup)"
    )


class GenerateStrategyRequest(BaseModel):
    """Request body for /generate-strategy endpoint."""
    deal_id: str = Field(..., description="Deal ID to generate strategy for")
    include_communication: bool = Field(
        True, description="Include communication recommendations"
    )


class SimulateScenarioRequest(BaseModel):
    """Request body for /simulate-scenario endpoint."""
    deal_id: str = Field(..., description="Deal ID to simulate on")
    action: str = Field(
        ..., description="Hypothetical action (e.g., 'schedule_demo', 'offer_discount')"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        None, description="Action parameters (e.g., discount_percent: 10)"
    )


# --- Response Models ---

class MLPrediction(BaseModel):
    """ML model prediction output."""
    win_probability: float
    confidence_lower: float
    confidence_upper: float
    confidence_width: float
    risk_score: float
    risk_level: str
    is_at_risk: bool
    low_confidence_warning: bool = False


class FeatureImportanceItem(BaseModel):
    """Single feature importance entry."""
    feature: str
    importance: float
    importance_pct: float


class AgentAnalysis(BaseModel):
    """Output from a single agent."""
    agent_name: str
    analysis: str
    confidence: float
    key_points: List[str]


class CriticReview(BaseModel):
    """Critic agent evaluation."""
    critique: str
    is_approved: bool
    issues: List[str]
    suggestions: List[str]


class HistoricalMatch(BaseModel):
    """A similar historical deal found by RAG."""
    deal_id: str
    outcome: str
    similarity: float
    engagement_per_week: Optional[float] = None
    silence_gap_days: Optional[float] = None
    duration_days: Optional[float] = None


class DealAnalysisResponse(BaseModel):
    """Full response from /analyze-deal."""
    deal_id: str
    predictions: MLPrediction
    lead_assessment: Optional[str] = None
    risk_analysis: Optional[str] = None
    strategy: Optional[str] = None
    communication: Optional[str] = None
    critic_review: Optional[str] = None
    historical_matches: List[HistoricalMatch] = []
    feature_importance: List[FeatureImportanceItem] = []


class PredictOutcomeResponse(BaseModel):
    """Response from /predict-outcome."""
    deal_id: Optional[str] = None
    win_probability: float
    confidence_lower: float
    confidence_upper: float
    confidence_width: float
    risk_score: float
    risk_level: str
    is_at_risk: bool
    low_confidence_warning: bool = False


class StrategyResponse(BaseModel):
    """Response from /generate-strategy."""
    deal_id: str
    strategy: str
    communication: Optional[str] = None
    risk_context: Optional[str] = None
    historical_evidence: List[HistoricalMatch] = []


class ScenarioResult(BaseModel):
    """Response from /simulate-scenario."""
    deal_id: str
    action: str
    current_win_probability: float
    simulated_win_probability: float
    probability_delta: float
    current_risk_score: float
    simulated_risk_score: float
    risk_delta: float
    explanation: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models_loaded: bool
    vector_store_ready: bool


class PipelineOverview(BaseModel):
    """Pipeline summary for dashboard."""
    total_deals: int
    deals_at_risk: int
    deals_healthy: int
    avg_win_probability: float
    avg_risk_score: float
    risk_distribution: Dict[str, int]
