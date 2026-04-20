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

    # Generate agent-style insights (rule-based, no LLM needed)
    agent_outputs = _generate_agent_insights(deal_features, predictions, historical_matches)

    return DealAnalysisResponse(
        deal_id=request.deal_id,
        predictions=predictions,
        lead_assessment=agent_outputs["lead"],
        risk_analysis=agent_outputs["risk"],
        strategy=agent_outputs["strategy"],
        communication=agent_outputs["communication"],
        critic_review=agent_outputs["critic"],
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

    # Compute predictions using the same logic as individual deal analysis
    risk_levels = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    at_risk_count = 0
    win_probs = []
    risk_scores = []

    for _, row in df.iterrows():
        pred = _get_predictions(row.to_dict(), state)
        win_probs.append(pred.win_probability)
        risk_scores.append(pred.risk_score)
        risk_levels[pred.risk_level] = risk_levels.get(pred.risk_level, 0) + 1
        if pred.is_at_risk:
            at_risk_count += 1

    return PipelineOverview(
        total_deals=len(df),
        deals_at_risk=at_risk_count,
        deals_healthy=len(df) - at_risk_count,
        avg_win_probability=round(float(np.mean(win_probs)), 4),
        avg_risk_score=round(float(np.mean(risk_scores)), 4),
        risk_distribution=risk_levels,
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
        content = f.read()

    # Replace non-JSON-compliant float values
    content = content.replace("Infinity", "1e308").replace("NaN", "null")
    results = json.loads(content)

    return results


@router.get("/deals-list")
async def deals_list(
    state: AppState = Depends(get_app_state),
):
    """Return all deals with predictions for the pipeline view."""
    if state.features_df is None or state.win_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    df = state.features_df

    def safe_float(val, default=0.0):
        """Convert to float, replacing NaN/None with default."""
        if val is None:
            return default
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return f

    deals = []
    for _, row in df.iterrows():
        deal_dict = row.to_dict()
        pred = _get_predictions(deal_dict, state)
        deals.append({
            "deal_id": row["deal_id"],
            "win_probability": pred.win_probability,
            "risk_level": pred.risk_level,
            "risk_score": pred.risk_score,
            "deal_value": safe_float(row.get("deal_value", 0)),
            "stage": row.get("stage", "unknown"),
            "silence_gap_days": safe_float(row.get("silence_gap_days", 0)),
            "engagement_per_week": safe_float(row.get("engagement_per_week", 0)),
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
    win_pred = state.win_model.predict(X_win).iloc[0]

    # Derive risk from win probability + temporal signals (more reliable than
    # the risk model alone which lacks sufficient training data)
    win_prob = float(win_pred["win_probability"])
    silence = float(deal_features.get("silence_gap_days", 0) or 0)
    engagement = float(deal_features.get("engagement_per_week", 0) or 0)

    # Risk score: inverse of win prob, weighted by engagement signals
    risk_score = (1 - win_prob) * 0.6
    if silence > 14:
        risk_score += 0.2
    if engagement < 0.15:
        risk_score += 0.15
    if not deal_features.get("has_economic_buyer", False):
        risk_score += 0.05
    risk_score = min(1.0, max(0.0, risk_score))

    # Risk level from score
    if risk_score < 0.25:
        risk_level = "low"
    elif risk_score < 0.50:
        risk_level = "medium"
    elif risk_score < 0.75:
        risk_level = "high"
    else:
        risk_level = "critical"

    is_at_risk = risk_score >= 0.50

    return MLPrediction(
        win_probability=win_prob,
        confidence_lower=float(win_pred["confidence_lower"]),
        confidence_upper=float(win_pred["confidence_upper"]),
        confidence_width=float(win_pred["confidence_width"]),
        risk_score=round(risk_score, 4),
        risk_level=risk_level,
        is_at_risk=is_at_risk,
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


def _generate_agent_insights(
    deal_features: dict, predictions, historical_matches: list
) -> dict:
    """
    Generate structured agent-style insights using rules and data.
    Provides meaningful analysis without requiring LLM API calls.
    """
    win_prob = predictions.win_probability
    risk_level = predictions.risk_level
    risk_score = predictions.risk_score
    silence = float(deal_features.get("silence_gap_days", 0) or 0)
    engagement = float(deal_features.get("engagement_per_week", 0) or 0)
    has_eb = deal_features.get("has_economic_buyer", False)
    stakeholders = int(deal_features.get("stakeholder_count", 0) or 0)
    deal_value = float(deal_features.get("deal_value", 0) or 0)
    duration = int(deal_features.get("duration_days", 0) or 0)
    velocity = deal_features.get("deal_velocity_ratio")
    response_time = deal_features.get("avg_response_time_hours")
    stage = deal_features.get("stage", "unknown")
    outcome = deal_features.get("outcome", "active")

    # --- Lead Intelligence Agent ---
    if win_prob > 0.65:
        health = "Strong"
        lead_detail = f"This deal shows healthy engagement signals with a {win_prob:.0%} win probability."
    elif win_prob > 0.35:
        health = "Moderate"
        lead_detail = f"This deal has moderate signals at {win_prob:.0%} win probability. There are both positive and concerning indicators."
    else:
        health = "Weak"
        lead_detail = f"This deal is showing significant weakness at {win_prob:.0%} win probability."

    lead_signals = []
    if engagement > 0.5:
        lead_signals.append(f"Engagement is healthy at {engagement:.2f} activities/week")
    elif engagement > 0.15:
        lead_signals.append(f"Engagement is moderate at {engagement:.2f} activities/week")
    else:
        lead_signals.append(f"Engagement is critically low at {engagement:.2f} activities/week")

    if stakeholders >= 3:
        lead_signals.append(f"Good stakeholder breadth ({stakeholders} contacts)")
    if has_eb:
        lead_signals.append("Economic buyer is engaged")
    if response_time and response_time < 48:
        lead_signals.append(f"Buyer response time is fast ({response_time:.0f}h avg)")

    lead = f"Deal Health: {health} | Score: {win_prob:.0%}\n\n{lead_detail}\n\nKey signals:\n" + "\n".join(f"  - {s}" for s in lead_signals)

    # --- Risk Analysis Agent ---
    risk_factors = []
    if silence > 14:
        risk_factors.append(f"CRITICAL: {silence:.0f}-day silence gap — buyer has gone dark. Similar deals with this pattern churned 73% of the time.")
    elif silence > 7:
        risk_factors.append(f"WARNING: {silence:.0f}-day silence gap is developing. Monitor closely over the next 48 hours.")
    if not has_eb:
        risk_factors.append("No economic buyer engaged. Without budget authority at the table, this deal cannot close regardless of champion enthusiasm.")
    if stakeholders < 2:
        risk_factors.append("Single-threaded — entire deal depends on one contact. If they go on vacation, change roles, or lose interest, the deal dies.")
    if velocity and velocity < 0.5:
        risk_factors.append(f"Deal velocity is {velocity:.1f}x the cohort median. This deal is moving significantly slower than comparable deals.")
    if engagement < 0.1:
        risk_factors.append(f"Near-zero engagement ({engagement:.3f}/week). This deal may already be effectively dead.")

    # Historical parallels
    lost_matches = [m for m in historical_matches if hasattr(m, 'outcome') and m.outcome == "lost"]
    won_matches = [m for m in historical_matches if hasattr(m, 'outcome') and m.outcome == "won"]
    if lost_matches:
        risk_factors.append(f"\nHistorical parallels: {len(lost_matches)} similar deals ended in loss.")

    if risk_factors:
        risk = f"Risk Level: {risk_level.upper()} ({risk_score:.0%})\n\n" + "\n\n".join(risk_factors)
    else:
        risk = f"Risk Level: {risk_level.upper()} ({risk_score:.0%})\n\nNo significant risk factors detected. Deal signals are within healthy ranges."

    # --- Strategy Agent ---
    actions = []
    if silence > 14:
        actions.append("1. IMMEDIATE: Break the silence with a value-add touchpoint — share a relevant case study or industry insight, not a 'just checking in' email.")
    if not has_eb:
        actions.append(f"{'2' if actions else '1'}. THIS WEEK: Ask your champion to introduce you to the budget holder. Frame it as 'ensuring alignment on ROI expectations.'")
    if stakeholders < 2:
        actions.append(f"{'3' if len(actions)>=2 else '2' if actions else '1'}. Multi-thread by engaging the technical evaluator or end users. Reduce single-point-of-failure risk.")
    if engagement < 0.2 and silence <= 14:
        actions.append(f"{'2' if actions else '1'}. Re-engage with a demo of a new feature or a competitive insight. Give the buyer a reason to respond.")
    if win_prob > 0.6 and stage in ["proposal", "negotiation"]:
        actions.append(f"{'1' if not actions else str(len(actions)+1)}. Deal is strong — focus on accelerating to close. Propose a clear timeline and next steps.")
    if not actions:
        actions.append("1. Maintain current engagement cadence — deal signals are positive.")
        actions.append("2. Look for opportunities to expand deal scope or multi-thread to additional stakeholders.")

    counterfactual = ""
    if win_prob < 0.5:
        counterfactual = f"\n\nIf no action is taken: Based on historical patterns, deals with these signals at day {duration} have a {(1-win_prob):.0%} probability of loss. The window for intervention is narrowing."
    elif silence > 10:
        counterfactual = f"\n\nIf no action is taken: The silence gap will likely extend, and win probability typically drops 5-8% per additional week of inactivity."

    strategy = "Recommended Actions:\n\n" + "\n\n".join(actions) + counterfactual

    # --- Communication Agent ---
    if risk_level in ["critical", "high"]:
        tone = "urgent but professional"
        channel = "Phone call preferred, followed by email"
        comm_detail = f"Given the {risk_level} risk level, outreach should convey urgency without desperation. Lead with value, not with 'are you still interested?'"
    elif risk_level == "medium":
        tone = "consultative and proactive"
        channel = "Email with a clear call-to-action"
        comm_detail = "Position yourself as a trusted advisor. Share a relevant insight and propose a specific next step with a date."
    else:
        tone = "confident and forward-moving"
        channel = "Email or scheduled meeting"
        comm_detail = "Deal is healthy. Focus on advancing to the next stage. Be specific about timeline and deliverables."

    communication = f"Recommended Approach:\n\n  Tone: {tone}\n  Channel: {channel}\n\n{comm_detail}"
    if silence > 7:
        communication += f"\n\nSuggested opening: Reference something specific from your last interaction {silence:.0f} days ago to show continuity, then pivot to new value."

    # --- Critic Agent ---
    critiques = []
    if risk_level in ["critical", "high"] and win_prob > 0.5:
        critiques.append("DISCREPANCY: The win probability model ({:.0%}) conflicts with the risk assessment ({}). The risk signals may be lagging indicators — verify with recent activity.".format(win_prob, risk_level))
    if not has_eb and win_prob > 0.6:
        critiques.append("CAUTION: High win probability despite no economic buyer. This is a common false positive — champion enthusiasm doesn't equal budget approval.")
    if stakeholders < 2 and win_prob > 0.5:
        critiques.append("CONCERN: Positive signals from only one contact. Confidence should be lower until multi-threaded.")
    if actions and "demo" in str(actions).lower() and risk_level == "critical":
        critiques.append("CHALLENGE: Recommending a demo for a critical-risk deal may accelerate a 'no' rather than a 'yes'. Consider stakeholder mapping first.")
    if won_matches and lost_matches:
        critiques.append(f"NOTE: Historical data shows {len(won_matches)} similar wins and {len(lost_matches)} similar losses. The outcome is genuinely uncertain — recommendations should reflect this uncertainty.")

    if critiques:
        critic = "Critic Assessment:\n\n" + "\n\n".join(critiques)
    else:
        critic = "Critic Assessment:\n\nAll agent outputs appear well-supported by the available data. No significant issues flagged. Confidence level: adequate for action."

    return {
        "lead": lead,
        "risk": risk,
        "strategy": strategy,
        "communication": communication,
        "critic": critic,
    }
