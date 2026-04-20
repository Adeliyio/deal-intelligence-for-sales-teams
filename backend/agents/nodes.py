"""
Agent node implementations for the LangGraph workflow.

Each node is a function that takes the current graph state,
runs an agent's analysis, and returns updated state.
"""
import json
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from backend.agents.agent_config import (
    AGENT_PROMPTS,
    AGENT_TEMPERATURES,
    MODEL_ROUTING,
)
from backend.agents.state import GraphState, AgentOutput, CritiqueOutput


def _get_llm(agent_name: str) -> ChatOpenAI:
    """Get the appropriate LLM for an agent based on routing config."""
    model = MODEL_ROUTING.get(agent_name, "openai")
    temperature = AGENT_TEMPERATURES.get(agent_name, 0.3)

    if model == "openai":
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
        )
    else:
        # Llama fallback — uses OpenAI-compatible API for local model
        return ChatOpenAI(
            model="llama-3.2",
            temperature=temperature,
            base_url="http://localhost:11434/v1",
            api_key="not-needed",
        )


def _format_deal_context(deal_context: Dict[str, Any]) -> str:
    """Format deal context into a readable prompt string."""
    features = deal_context.get("deal_features", {})
    predictions = deal_context.get("ml_predictions", {})
    history = deal_context.get("historical_matches", [])

    context = f"""## Deal: {deal_context.get('deal_id', 'Unknown')}

### ML Predictions
- Win Probability: {predictions.get('win_probability', 'N/A')}
- Confidence Interval: [{predictions.get('confidence_lower', 'N/A')}, {predictions.get('confidence_upper', 'N/A')}]
- Risk Score: {predictions.get('risk_score', 'N/A')}
- Risk Level: {predictions.get('risk_level', 'N/A')}

### Key Features
- Deal Value: ${features.get('deal_value', 0):,.0f}
- Stage: {features.get('stage', 'Unknown')}
- Duration: {features.get('duration_days', 0)} days
- Engagement Score: {features.get('engagement_score', 0):.2f}
- Engagement/Week: {features.get('engagement_per_week', 0):.3f}
- Response Time (avg): {features.get('avg_response_time_hours', 'N/A')} hours
- Silence Gap: {features.get('silence_gap_days', 0)} days
- Silence Severity: {features.get('silence_gap_severity', 0):.2f}
- Stakeholder Count: {features.get('stakeholder_count', 0)}
- Has Economic Buyer: {features.get('has_economic_buyer', False)}
- Decision Maker Ratio: {features.get('decision_maker_ratio', 0):.2f}
- Deal Velocity Ratio: {features.get('deal_velocity_ratio', 'N/A')}
- Decay-Weighted Engagement: {features.get('decay_weighted_engagement', 0):.3f}
"""

    if history:
        context += "\n### Similar Historical Deals\n"
        for i, h in enumerate(history[:3], 1):
            context += f"{i}. Deal {h.get('deal_id', '?')} — Outcome: {h.get('outcome', '?')}, "
            context += f"Similarity: {h.get('similarity', 0):.2f}\n"

    return context


def _parse_agent_response(response: str, agent_name: str) -> AgentOutput:
    """Parse LLM response into structured AgentOutput."""
    # Extract key points (lines starting with - or numbered)
    lines = response.strip().split("\n")
    key_points = [
        line.strip("- •").strip()
        for line in lines
        if line.strip().startswith(("-", "•", "1", "2", "3"))
    ][:5]

    return AgentOutput(
        agent_name=agent_name,
        analysis=response,
        confidence=0.7,  # Default; refined by critic
        key_points=key_points,
        evidence=[],
    )


def lead_intelligence_node(state: GraphState) -> GraphState:
    """Lead Intelligence Agent: synthesizes ML output into deal assessment."""
    llm = _get_llm("lead_intelligence")
    context = _format_deal_context(state["deal_context"])

    messages = [
        SystemMessage(content=AGENT_PROMPTS["lead_intelligence"]),
        HumanMessage(content=f"Analyze this deal:\n\n{context}"),
    ]

    response = llm.invoke(messages)
    state["lead_output"] = _parse_agent_response(
        response.content, "lead_intelligence"
    )

    return state


def risk_analysis_node(state: GraphState) -> GraphState:
    """Risk Analysis Agent: identifies failure patterns and warning signals."""
    llm = _get_llm("risk_analysis")
    context = _format_deal_context(state["deal_context"])

    # Include lead output if available for context
    additional = ""
    if state.get("lead_output"):
        additional = f"\n\n### Lead Agent Assessment\n{state['lead_output']['analysis']}"

    messages = [
        SystemMessage(content=AGENT_PROMPTS["risk_analysis"]),
        HumanMessage(
            content=f"Analyze risk for this deal:\n\n{context}{additional}"
        ),
    ]

    response = llm.invoke(messages)
    state["risk_output"] = _parse_agent_response(
        response.content, "risk_analysis"
    )

    return state


def strategy_node(state: GraphState) -> GraphState:
    """Strategy Agent: recommends actions and simulates scenarios."""
    llm = _get_llm("strategy")
    context = _format_deal_context(state["deal_context"])

    # Include risk output for strategy recommendations
    additional = ""
    if state.get("risk_output"):
        additional = f"\n\n### Risk Assessment\n{state['risk_output']['analysis']}"

    messages = [
        SystemMessage(content=AGENT_PROMPTS["strategy"]),
        HumanMessage(
            content=f"Recommend strategy for this deal:\n\n{context}{additional}"
        ),
    ]

    response = llm.invoke(messages)
    state["strategy_output"] = _parse_agent_response(
        response.content, "strategy"
    )

    return state


def communication_node(state: GraphState) -> GraphState:
    """Communication Agent: generates adaptive outreach recommendations."""
    llm = _get_llm("communication")
    context = _format_deal_context(state["deal_context"])

    # Include strategy for communication guidance
    additional = ""
    if state.get("strategy_output"):
        additional = (
            f"\n\n### Strategy Recommendations\n"
            f"{state['strategy_output']['analysis']}"
        )

    messages = [
        SystemMessage(content=AGENT_PROMPTS["communication"]),
        HumanMessage(
            content=f"Draft communication approach for this deal:\n\n{context}{additional}"
        ),
    ]

    response = llm.invoke(messages)
    state["communication_output"] = _parse_agent_response(
        response.content, "communication"
    )

    return state


def critic_node(state: GraphState) -> GraphState:
    """Critic Agent: challenges all other agents' outputs."""
    llm = _get_llm("critic")
    context = _format_deal_context(state["deal_context"])

    # Compile all agent outputs for review
    outputs_to_review = ""
    for agent_key in ["lead_output", "risk_output", "strategy_output", "communication_output"]:
        output = state.get(agent_key)
        if output:
            outputs_to_review += f"\n\n### {output['agent_name'].replace('_', ' ').title()} Output\n"
            outputs_to_review += output["analysis"]

    messages = [
        SystemMessage(content=AGENT_PROMPTS["critic"]),
        HumanMessage(
            content=(
                f"Review and critique the following agent outputs for this deal:\n\n"
                f"{context}\n\n---\n\n"
                f"## Agent Outputs to Evaluate{outputs_to_review}"
            )
        ),
    ]

    response = llm.invoke(messages)

    critique = CritiqueOutput(
        target_agent="all",
        is_approved=True,  # Default; parsed from response
        critique=response.content,
        issues=[],
        suggestions=[],
    )

    if state.get("critiques") is None:
        state["critiques"] = []
    state["critiques"].append(critique)

    return state


def synthesize_node(state: GraphState) -> GraphState:
    """Synthesize all agent outputs into a final report."""
    report = {
        "deal_id": state["deal_context"]["deal_id"],
        "lead_assessment": (
            state["lead_output"]["analysis"] if state.get("lead_output") else None
        ),
        "risk_analysis": (
            state["risk_output"]["analysis"] if state.get("risk_output") else None
        ),
        "strategy": (
            state["strategy_output"]["analysis"]
            if state.get("strategy_output")
            else None
        ),
        "communication": (
            state["communication_output"]["analysis"]
            if state.get("communication_output")
            else None
        ),
        "critic_review": (
            state["critiques"][-1]["critique"]
            if state.get("critiques")
            else None
        ),
        "ml_predictions": state["deal_context"].get("ml_predictions", {}),
    }

    state["final_report"] = report
    return state
