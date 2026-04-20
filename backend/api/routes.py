"""
API route implementations for deal intelligence endpoints.
"""
import os
import json
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any

from backend.api.schemas import (
    DealAnalysisRequest,
    DealAnalysisResponse,
    PredictOutcomeRequest,
    PredictOutcomeResponse,
    GenerateStrategyRequest,
    StrategyResponse,
    SimulateScenarioRequest,
    ScenarioResult,
    MLPrediction,
    HistoricalMatch,
    FeatureImportanceItem,
    PipelineOverview,
)
from backend.api.dependencies import AppState, get_app_state
from backend.ml.preprocessing import prepare_inference_data
from backend.ml.model_config import WIN_MODEL_FEATURES, RISK_MODEL_FEATURES

router = APIRouter()


@router.post("/analyze-deal", response_model=DealAnalysisResponse)
async def analyze_deal(
    request: DealAnalysisRequest,
    state: AppState = Depends(get_app_state),
):
    """
    Full deal analysis: ML predictions + agent reasoning + historical context.

    This is the primary endpoint — combines all system components into
    a single comprehensive analysis.
    """
    # Look up deal features
    deal_features = state.get_deal_features(request.deal_id)
    if not deal_features:
        raise HTTPException(status_code=404, detail=f"Deal {request.deal_id} not found")

    # ML predictions
    predictions = _get_predictions(deal_features, state)

    # Historical matches from RAG
    historical_matches = []
    if state.retriever:
        similar = state.retriever.store.search(deal_features, top_k=5)
        historical_matches = [
            HistoricalMatch(
                deal_id=d.get("deal_id", ""),
                outcome=d.get("outcome", ""),
                similarity=d.get("similarity", 0),
                engagement_per_week=d.get("engagement_per_week"),
                silence_gap_days=d.get("silence_gap_days"),
                duration_days=d.get("duration_days"),
            )
            for d in similar
            if d.get("deal_id") != request.deal_id  # Exclude self
        ]

    # Feature importance
    feature_importance = []
    if state.win_model:
        fi = state.win_model.get_feature_importance(WIN_MODEL_FEATURES)
        feature_importance = [
            FeatureImportanceItem(
                feature=row["feature"],
                importance=row["importance"],
                importance_pct=row["importance_pct"],
            )
            for _, row in fi.head(10).iterrows()
        ]

    return DealAnalysisResponse(
        deal_id=request.deal_id,
        predictions=predictions,
        historical_matches=historical_matches,
        feature_importance=feature_importance,
    )


@router.post("/predict-outcome", response_model=PredictOutcomeResponse)
async def predict_outcome(
    request: PredictOutcomeRequest,
    state: AppState = Depends(get_app_state),
):
    """
    Predict win probability and risk score for a deal.

    Accepts either a deal_id (looks up features) or raw features dict.
    Returns calibrated probability with confidence intervals.
    """
    if request.deal_id:
        deal_features = state.get_deal_features(request.deal_id)
        if not deal_features:
            raise HTTPException(
                status_code=404, detail=f"Deal {request.deal_id} not found"
            )
    elif request.features:
        deal_features = request.features
    else:
        raise HTTPException(
            status_code=400, detail="Provide either deal_id or features"
        )

    predictions = _get_predictions(deal_features, state)

    return PredictOutcomeResponse(
        deal_id=request.deal_id,
        win_probability=predictions.win_probability,
        confidence_lower=predictions.confidence_lower,
        confidence_upper=predictions.confidence_upper,
        confidence_width=predictions.confidence_width,
        risk_score=predictions.risk_score,
        risk_level=predictions.risk_level,
        is_at_risk=predictions.is_at_risk,
        low_confidence_warning=predictions.low_confidence_warning,
    )


@router.post("/generate-strategy", response_model=StrategyResponse)
async def generate_strategy(
    request: GenerateStrategyRequest,
    state: AppState = Depends(get_app_state),
):
    """
    Generate strategic recommendations for a deal.

    Uses risk context and historical evidence to ground recommendations.
    """
    deal_features = state.get_deal_features(request.deal_id)
    if not deal_features:
        raise HTTPException(status_code=404, detail=f"Deal {request.deal_id} not found")

    # Get historical evidence for strategy
    historical_evidence = []
    risk_context = None

    if state.retriever:
        won_deals = state.retriever.find_successful_strategies(deal_features)
        historical_evidence = [
            HistoricalMatch(
                deal_id=d.get("deal_id", ""),
                outcome="won",
                similarity=d.get("similarity", 0),
                engagement_per_week=d.get("engagement_per_week"),
                duration_days=d.get("duration_days"),
            )
            for d in won_deals
        ]

        # RAG confidence check
        check = state.retriever.graceful_degradation_check(deal_features)
        if not check.get("has_sufficient_context"):
            risk_context = check.get("reason", "Limited historical data available")

    # Build strategy summary from deal state
    strategy = _build_strategy_summary(deal_features, state)

    return StrategyResponse(
        deal_id=request.deal_id,
        strategy=strategy,
        risk_context=risk_context,
        historical_evidence=historical_evidence,
    )


@router.post("/simulate-scenario", response_model=ScenarioResult)
async def simulate_scenario(
    request: SimulateScenarioRequest,
    state: AppState = Depends(get_app_state),
):
    """
    Simulate the impact of a hypothetical action on deal outcomes.

    Models counterfactuals: "what happens to win probability if we
    schedule a demo this week?" vs "what if we offer a 10% discount?"
    """
    deal_features = state.get_deal_features(request.deal_id)
    if not deal_features:
        raise HTTPException(status_code=404, detail=f"Deal {request.deal_id} not found")

    # Current predictions
    current_predictions = _get_predictions(deal_features, state)

    # Simulate action by modifying features
    simulated_features = _simulate_action(
        deal_features.copy(), request.action, request.parameters
    )

    # Get simulated predictions
    simulated_predictions = _get_predictions(simulated_features, state)

    # Compute deltas
    prob_delta = simulated_predictions.win_probability - current_predictions.win_probability
    risk_delta = simulated_predictions.risk_score - current_predictions.risk_score

    explanation = _explain_simulation(
        request.action, prob_delta, risk_delta, request.parameters
    )

    return ScenarioResult(
        deal_id=request.deal_id,
        action=request.action,
        current_win_probability=current_predictions.win_probability,
        simulated_win_probability=simulated_predictions.win_probability,
        probability_delta=round(prob_delta, 4),
        current_risk_score=current_predictions.risk_score,
        simulated_risk_score=simulated_predictions.risk_score,
        risk_delta=round(risk_delta, 4),
        explanation=explanation,
    )


@router.get("/pipeline-overview", response_model=PipelineOverview)
async def pipeline_overview(
    state: AppState = Depends(get_app_state),
):
    """Get a summary of the entire deal pipeline."""
    if state.features_df is None or state.win_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    df = state.features_df
    X_win = prepare_inference_data(df, model_type="win")
    X_risk = prepare_inference_data(df, model_type="risk")

    win_preds = state.win_model.predict(X_win)
    risk_preds = state.risk_model.predict(X_risk)

    risk_counts = risk_preds["risk_level"].value_counts().to_dict()

    return PipelineOverview(
        total_deals=len(df),
        deals_at_risk=int(risk_preds["is_at_risk"].sum()),
        deals_healthy=int((~risk_preds["is_at_risk"].astype(bool)).sum()),
        avg_win_probability=round(float(win_preds["win_probability"].mean()), 4),
        avg_risk_score=round(float(risk_preds["risk_score"].mean()), 4),
        risk_distribution={str(k): int(v) for k, v in risk_counts.items()},
    )


@router.get("/evaluation-results")
async def evaluation_results():
    """
    Return pre-computed evaluation metrics.
    Includes model performance, calibration analysis, and Critic A/B test results.
    """
    eval_path = "models/saved/evaluation_results.json"
    if not os.path.exists(eval_path):
        raise HTTPException(status_code=404, detail="Evaluation results not found. Run scripts/run_evaluation.py first.")

    with open(eval_path, "r") as f:
        results = json.load(f)

    return results


@router.get("/deals-list")
async def deals_list(
    state: AppState = Depends(get_app_state),
):
    """Return all deals with predictions for the pipeline view."""
    if state.features_df is None or state.win_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    df = state.features_df
    X_win = prepare_inference_data(df, model_type="win")
    X_risk = prepare_inference_data(df, model_type="risk")

    win_preds = state.win_model.predict(X_win)
    risk_preds = state.risk_model.predict(X_risk)

    deals = []
    for i, (_, row) in enumerate(df.iterrows()):
        deals.append({
            "deal_id": row["deal_id"],
            "win_probability": float(win_preds.iloc[i]["win_probability"]),
            "risk_level": str(risk_preds.iloc[i]["risk_level"]),
            "risk_score": float(risk_preds.iloc[i]["risk_score"]),
            "deal_value": float(row.get("deal_value", 0)),
            "stage": row.get("stage", "unknown"),
            "silence_gap_days": float(row.get("silence_gap_days", 0) or 0),
            "engagement_per_week": float(row.get("engagement_per_week", 0) or 0),
            "industry": row.get("industry", ""),
            "deal_size_bucket": row.get("deal_size_bucket", ""),
            "outcome": row.get("outcome", ""),
        })

    return {"deals": deals}


# --- Helper functions ---


def _get_predictions(deal_features: dict, state: AppState) -> MLPrediction:
    """Run both models on a deal's features."""
    if not state.win_model or not state.risk_model:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Prepare features as single-row DataFrame
    df = pd.DataFrame([deal_features])

    if "has_economic_buyer" in df.columns:
        df["has_economic_buyer"] = df["has_economic_buyer"].astype(int)

    X_win = prepare_inference_data(df, model_type="win")
    X_risk = prepare_inference_data(df, model_type="risk")

    win_pred = state.win_model.predict(X_win).iloc[0]
    risk_pred = state.risk_model.predict(X_risk).iloc[0]

    return MLPrediction(
        win_probability=float(win_pred["win_probability"]),
        confidence_lower=float(win_pred["confidence_lower"]),
        confidence_upper=float(win_pred["confidence_upper"]),
        confidence_width=float(win_pred["confidence_width"]),
        risk_score=float(risk_pred["risk_score"]),
        risk_level=str(risk_pred["risk_level"]),
        is_at_risk=bool(risk_pred["is_at_risk"]),
        low_confidence_warning=bool(win_pred["low_confidence_warning"]),
    )


def _simulate_action(
    features: dict, action: str, parameters: dict = None
) -> dict:
    """
    Modify features to simulate the effect of an action.

    This is a simplified simulation — in production, these would be
    learned from historical intervention data.
    """
    params = parameters or {}

    if action == "schedule_demo":
        # Demo increases engagement, reduces silence
        features["engagement_score"] = features.get("engagement_score", 0) + 0.4
        features["engagement_per_week"] = features.get("engagement_per_week", 0) + 0.1
        features["silence_gap_days"] = max(0, features.get("silence_gap_days", 0) - 7)
        features["total_activities"] = features.get("total_activities", 0) + 2

    elif action == "executive_outreach":
        # Stakeholder coverage improves
        features["stakeholder_count"] = features.get("stakeholder_count", 0) + 1
        features["has_economic_buyer"] = 1
        features["decision_maker_ratio"] = min(
            1.0, features.get("decision_maker_ratio", 0) + 0.2
        )

    elif action == "offer_discount":
        # May accelerate but doesn't change engagement
        discount = params.get("discount_percent", 10)
        features["deal_value"] = features.get("deal_value", 0) * (1 - discount / 100)
        features["silence_gap_days"] = max(0, features.get("silence_gap_days", 0) - 3)

    elif action == "send_followup":
        # Mild engagement bump, slight silence reduction
        features["total_activities"] = features.get("total_activities", 0) + 1
        features["silence_gap_days"] = max(0, features.get("silence_gap_days", 0) - 2)
        features["decay_weighted_engagement"] = (
            features.get("decay_weighted_engagement", 0) + 0.15
        )

    elif action == "stakeholder_mapping":
        # Multi-threading improvement
        features["stakeholder_count"] = features.get("stakeholder_count", 0) + 2
        features["stakeholder_breadth"] = features.get("stakeholder_breadth", 0) + 1
        features["decision_maker_ratio"] = min(
            1.0, features.get("decision_maker_ratio", 0) + 0.15
        )

    return features


def _explain_simulation(
    action: str, prob_delta: float, risk_delta: float, parameters: dict = None
) -> str:
    """Generate human-readable explanation of simulation results."""
    direction = "increase" if prob_delta > 0 else "decrease"
    risk_direction = "increase" if risk_delta > 0 else "decrease"

    explanation = (
        f"Simulating '{action}' suggests a {abs(prob_delta):.1%} {direction} "
        f"in win probability and a {abs(risk_delta):.1%} {risk_direction} in risk score."
    )

    if prob_delta > 0.05:
        explanation += " This action appears high-impact for this deal."
    elif prob_delta > 0:
        explanation += " Modest positive impact expected."
    else:
        explanation += " Limited or negative impact — consider alternatives."

    return explanation


def _build_strategy_summary(deal_features: dict, state: AppState) -> str:
    """Build a rule-based strategy summary when LLM agents are not invoked."""
    signals = []

    silence = deal_features.get("silence_gap_days", 0) or 0
    engagement = deal_features.get("engagement_per_week", 0) or 0
    has_eb = deal_features.get("has_economic_buyer", False)
    stakeholders = deal_features.get("stakeholder_count", 0) or 0
    velocity = deal_features.get("deal_velocity_ratio")

    if silence > 14:
        signals.append(f"URGENT: {silence:.0f}-day silence gap detected. Immediate outreach recommended.")
    if engagement < 0.2:
        signals.append("Low engagement rate. Consider re-engagement campaign or value reinforcement.")
    if not has_eb:
        signals.append("No economic buyer identified. Prioritize executive stakeholder mapping.")
    if stakeholders < 2:
        signals.append("Single-threaded deal. Risk of champion loss. Multi-thread immediately.")
    if velocity and velocity < 0.5:
        signals.append("Deal moving slower than cohort median. Identify and address blockers.")

    if not signals:
        signals.append("Deal appears healthy. Maintain current engagement cadence.")

    return "\n".join(signals)
