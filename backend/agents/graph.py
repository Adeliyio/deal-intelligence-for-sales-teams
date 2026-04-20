"""
LangGraph workflow definition for the multi-agent deal analysis system.

The agents collaborate in a structured graph:
  Lead Agent → Risk Agent → Strategy Agent → Communication Agent → Critic → Synthesize

The Critic evaluates all outputs before the final report is generated.
This is not a sequential pipeline — the Critic can trigger revision.
"""
from langgraph.graph import StateGraph, END
from backend.agents.state import GraphState
from backend.agents.nodes import (
    lead_intelligence_node,
    risk_analysis_node,
    strategy_node,
    communication_node,
    critic_node,
    synthesize_node,
)
from backend.agents.agent_config import MAX_DEBATE_ROUNDS


def should_revise(state: GraphState) -> str:
    """
    Determine if the Critic's feedback requires another round.
    Routes to either revision or final synthesis.
    """
    critiques = state.get("critiques", [])
    debate_round = state.get("debate_round", 0)

    # Don't exceed max debate rounds
    if debate_round >= MAX_DEBATE_ROUNDS:
        return "synthesize"

    # Check if critic flagged major issues
    if critiques:
        latest = critiques[-1]
        if not latest.get("is_approved", True) and latest.get("issues"):
            return "revise"

    return "synthesize"


def build_agent_graph() -> StateGraph:
    """
    Build the multi-agent LangGraph workflow.

    Graph structure:
        lead → risk → strategy → communication → critic → [revise|synthesize]
    """
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("lead_intelligence", lead_intelligence_node)
    workflow.add_node("risk_analysis", risk_analysis_node)
    workflow.add_node("strategy", strategy_node)
    workflow.add_node("communication", communication_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("synthesize", synthesize_node)

    # Define edges — sequential flow with critic gate
    workflow.set_entry_point("lead_intelligence")
    workflow.add_edge("lead_intelligence", "risk_analysis")
    workflow.add_edge("risk_analysis", "strategy")
    workflow.add_edge("strategy", "communication")
    workflow.add_edge("communication", "critic")

    # Conditional edge: critic decides if revision needed
    workflow.add_conditional_edges(
        "critic",
        should_revise,
        {
            "revise": "risk_analysis",  # Re-run from risk with critic feedback
            "synthesize": "synthesize",
        },
    )

    workflow.add_edge("synthesize", END)

    return workflow


def compile_agent_graph():
    """Compile the graph into an executable workflow."""
    workflow = build_agent_graph()
    return workflow.compile()


def run_deal_analysis(deal_context: dict) -> dict:
    """
    Run the full multi-agent analysis on a deal.

    Args:
        deal_context: Dict with deal_id, deal_features, ml_predictions,
                     and historical_matches.

    Returns:
        Final report dict with all agent outputs and critic review.
    """
    graph = compile_agent_graph()

    initial_state = GraphState(
        deal_context=deal_context,
        lead_output=None,
        risk_output=None,
        strategy_output=None,
        communication_output=None,
        critiques=[],
        final_report=None,
        debate_round=0,
        errors=[],
    )

    result = graph.invoke(initial_state)

    return result.get("final_report", {})
