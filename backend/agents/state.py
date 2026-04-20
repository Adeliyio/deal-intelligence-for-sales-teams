"""
Agent state definitions for the LangGraph workflow.

The state flows through the multi-agent graph, accumulating outputs
from each agent as they analyze the deal.
"""
from typing import TypedDict, Optional, List, Dict, Any


class DealContext(TypedDict):
    """Input context for the agent system."""
    deal_id: str
    deal_features: Dict[str, Any]
    ml_predictions: Dict[str, Any]
    historical_matches: List[Dict[str, Any]]


class AgentOutput(TypedDict):
    """Structured output from a single agent."""
    agent_name: str
    analysis: str
    confidence: float
    key_points: List[str]
    evidence: List[str]


class CritiqueOutput(TypedDict):
    """Output from the Critic Agent evaluating another agent."""
    target_agent: str
    is_approved: bool
    critique: str
    issues: List[str]
    suggestions: List[str]


class GraphState(TypedDict):
    """Full state flowing through the LangGraph agent graph."""
    # Input
    deal_context: DealContext

    # Agent outputs (accumulated as graph executes)
    lead_output: Optional[AgentOutput]
    risk_output: Optional[AgentOutput]
    strategy_output: Optional[AgentOutput]
    communication_output: Optional[AgentOutput]

    # Critic evaluations
    critiques: List[CritiqueOutput]

    # Final synthesized output
    final_report: Optional[Dict[str, Any]]

    # Metadata
    debate_round: int
    errors: List[str]
