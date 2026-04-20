"""
Retrieval interface for the agent system.

Provides structured query methods that agents use to ground their
analysis in historical deal evidence. This is what prevents hallucination —
every claim about "deals like this" is backed by actual retrieved records.
"""
import pandas as pd
from typing import Dict, List, Optional

from backend.rag.vector_store import DealVectorStore


class DealRetriever:
    """
    High-level retrieval interface used by agents.

    Wraps the vector store with query patterns specific to deal analysis:
    - Outcome-filtered similarity search
    - Silence pattern matching
    - Cohort comparison
    - Evidence gathering for risk claims
    """

    def __init__(self, vector_store: DealVectorStore):
        self.store = vector_store

    @classmethod
    def from_features(cls, features_path: str = "data/processed/deal_features.csv"):
        """Build retriever from saved feature matrix."""
        df = pd.read_csv(features_path)

        # Convert booleans
        if "has_economic_buyer" in df.columns:
            df["has_economic_buyer"] = df["has_economic_buyer"].astype(int)

        store = DealVectorStore(embedding_mode="feature")
        store.build_index(df)

        return cls(store)

    def get_deal_context(
        self, deal_features: dict, top_k: int = 5
    ) -> Dict:
        """
        Get full retrieval context for a deal — used by the agent system.

        Returns a structured context dict with:
        - similar_won: deals like this that succeeded
        - similar_lost: deals like this that failed
        - silence_matches: deals with similar silence patterns
        - cohort_insights: aggregate stats for this deal's cohort
        """
        context = {
            "similar_won": self.store.find_similar_won_deals(
                deal_features, top_k=min(3, top_k)
            ),
            "similar_lost": self.store.find_similar_lost_deals(
                deal_features, top_k=min(3, top_k)
            ),
            "silence_matches": [],
            "cohort_insights": {},
        }

        # Silence pattern matching
        silence_days = deal_features.get("silence_gap_days", 0)
        duration = deal_features.get("duration_days", 0)
        if silence_days and duration:
            context["silence_matches"] = self.store.find_silence_pattern_matches(
                silence_gap_days=silence_days,
                duration_days=duration,
                top_k=3,
            )

        # Cohort insights
        industry = deal_features.get("industry")
        size_bucket = deal_features.get("deal_size_bucket")
        if industry and size_bucket:
            context["cohort_insights"] = self.store.get_cohort_insights(
                industry=industry, deal_size_bucket=size_bucket
            )

        return context

    def find_evidence_for_risk(
        self, deal_features: dict, risk_factors: List[str]
    ) -> List[Dict]:
        """
        Find historical evidence supporting specific risk claims.

        Used by the Risk Agent to ground its analysis:
        "3 similar deals with this silence pattern churned within 2 weeks"
        """
        evidence = []

        # Find lost deals with similar profiles
        similar_lost = self.store.find_similar_lost_deals(
            deal_features, top_k=5
        )

        for deal in similar_lost:
            evidence_item = {
                "deal_id": deal.get("deal_id"),
                "outcome": deal.get("outcome"),
                "similarity": deal.get("similarity"),
                "matching_signals": [],
            }

            # Check which risk factors this historical deal shares
            for factor in risk_factors:
                if factor == "high_silence" and deal.get("silence_gap_days", 0) > 14:
                    evidence_item["matching_signals"].append(
                        f"Silence gap: {deal['silence_gap_days']:.0f} days"
                    )
                elif factor == "low_engagement" and deal.get("engagement_per_week", 1) < 0.2:
                    evidence_item["matching_signals"].append(
                        f"Low engagement: {deal['engagement_per_week']:.3f}/week"
                    )
                elif factor == "no_economic_buyer" and not deal.get("has_economic_buyer"):
                    evidence_item["matching_signals"].append(
                        "No economic buyer engaged"
                    )
                elif factor == "slow_velocity" and deal.get("deal_velocity_ratio", 1) < 0.5:
                    evidence_item["matching_signals"].append(
                        f"Slow velocity: {deal['deal_velocity_ratio']:.2f}x cohort"
                    )

            if evidence_item["matching_signals"]:
                evidence.append(evidence_item)

        return evidence

    def find_successful_strategies(
        self, deal_features: dict
    ) -> List[Dict]:
        """
        Find what successful reps did with similar deals.

        Used by the Strategy Agent to ground recommendations in
        evidence of what actually worked historically.
        """
        won_deals = self.store.find_similar_won_deals(deal_features, top_k=5)

        strategies = []
        for deal in won_deals:
            strategy = {
                "deal_id": deal.get("deal_id"),
                "similarity": deal.get("similarity"),
                "engagement_per_week": deal.get("engagement_per_week"),
                "stakeholder_count": deal.get("stakeholder_count"),
                "has_economic_buyer": deal.get("has_economic_buyer"),
                "duration_days": deal.get("duration_days"),
                "deal_velocity_ratio": deal.get("deal_velocity_ratio"),
            }
            strategies.append(strategy)

        return strategies

    def graceful_degradation_check(self, deal_features: dict) -> Dict:
        """
        Check if the RAG layer has enough data to provide useful context.

        When no similar past deals exist, the system says so explicitly
        rather than hallucinating a pattern. This is the cold-start
        graceful degradation behavior.
        """
        all_results = self.store.search(deal_features, top_k=3)

        if not all_results:
            return {
                "has_sufficient_context": False,
                "reason": "No similar historical deals found in the vector store",
                "recommendation": "Rely on ML model output and heuristics only",
            }

        avg_similarity = sum(r["similarity"] for r in all_results) / len(all_results)

        if avg_similarity < 0.5:
            return {
                "has_sufficient_context": False,
                "reason": f"Best matches have low similarity (avg: {avg_similarity:.2f})",
                "recommendation": "Historical comparisons may be unreliable",
            }

        return {
            "has_sufficient_context": True,
            "avg_similarity": round(avg_similarity, 3),
            "n_matches": len(all_results),
        }
