"""
Vector store for historical deal retrieval.

Uses FAISS for local similarity search. Indexes deal records by their
feature embeddings so the system can answer queries like:
- "Find deals that looked like this at day 30 and ultimately closed"
- "Find deals with this silence gap pattern that churned"
- "What did successful reps do differently with this buyer profile?"

Every risk assessment is grounded in retrieved evidence, not generated
from thin air — this eliminates hallucination on deal-specific claims.
"""
import os
import json
import numpy as np
import pandas as pd
import faiss
from typing import List, Dict, Optional

from backend.rag.rag_config import (
    FAISS_INDEX_PATH,
    TOP_K_RESULTS,
    SIMILARITY_THRESHOLD,
)
from backend.rag.embeddings import DealEmbedder


class DealVectorStore:
    """
    FAISS-based vector store for historical deal retrieval.

    Stores deal embeddings and metadata, enabling similarity search
    to find historical parallels for current deals.
    """

    def __init__(self, embedding_mode: str = "feature"):
        self.embedder = DealEmbedder(mode=embedding_mode)
        self.index = None
        self.metadata = []  # Deal metadata aligned with index positions
        self.is_built = False

    def build_index(self, features_df: pd.DataFrame) -> None:
        """
        Build the FAISS index from a feature matrix.

        Args:
            features_df: DataFrame with deal features (output of TemporalFeatureEngineer)
        """
        # Generate embeddings
        embeddings = self.embedder.embed_deals(features_df)
        dimension = embeddings.shape[1]

        # Normalize for cosine similarity (FAISS inner product on normalized = cosine)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = embeddings / norms

        # Build FAISS index (Inner Product = cosine on normalized vectors)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(normalized.astype(np.float32))

        # Store metadata for retrieval
        self.metadata = features_df.to_dict("records")
        self.is_built = True

    def search(
        self,
        query_features: dict,
        top_k: int = TOP_K_RESULTS,
        filter_outcome: Optional[str] = None,
    ) -> List[Dict]:
        """
        Find similar historical deals.

        Args:
            query_features: dict of feature values for the query deal
            top_k: number of results to return
            filter_outcome: optionally filter to only "won" or "lost" deals

        Returns:
            List of dicts with deal metadata and similarity scores
        """
        if not self.is_built:
            raise RuntimeError("Index not built. Call build_index() first.")

        # Generate query embedding
        query_vector = self.embedder.embed_query(query_features)

        # Normalize query
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm

        query_vector = query_vector.astype(np.float32).reshape(1, -1)

        # Search (retrieve extra if filtering)
        search_k = top_k * 3 if filter_outcome else top_k
        scores, indices = self.index.search(query_vector, min(search_k, len(self.metadata)))

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for empty slots
                continue

            deal = self.metadata[idx].copy()
            similarity = float(score)

            # Apply threshold
            if similarity < SIMILARITY_THRESHOLD:
                continue

            # Apply outcome filter
            if filter_outcome and deal.get("outcome") != filter_outcome:
                continue

            deal["similarity"] = round(similarity, 4)
            results.append(deal)

            if len(results) >= top_k:
                break

        return results

    def find_similar_won_deals(
        self, query_features: dict, top_k: int = 3
    ) -> List[Dict]:
        """Find historical deals similar to this one that were won."""
        return self.search(query_features, top_k=top_k, filter_outcome="won")

    def find_similar_lost_deals(
        self, query_features: dict, top_k: int = 3
    ) -> List[Dict]:
        """Find historical deals similar to this one that were lost."""
        return self.search(query_features, top_k=top_k, filter_outcome="lost")

    def find_silence_pattern_matches(
        self, silence_gap_days: float, duration_days: float, top_k: int = 5
    ) -> List[Dict]:
        """
        Find deals with similar silence patterns.
        Useful for answering: "What happened to deals that went quiet like this?"
        """
        if not self.is_built:
            raise RuntimeError("Index not built.")

        matches = []
        for deal in self.metadata:
            deal_silence = deal.get("silence_gap_days", 0) or 0
            deal_duration = deal.get("duration_days", 0) or 0

            # Similarity based on relative silence (gap/duration ratio)
            if deal_duration > 0 and duration_days > 0:
                query_ratio = silence_gap_days / duration_days
                deal_ratio = deal_silence / deal_duration
                diff = abs(query_ratio - deal_ratio)

                if diff < 0.3:  # Within 30% relative difference
                    deal_copy = deal.copy()
                    deal_copy["pattern_similarity"] = round(1 - diff, 3)
                    matches.append(deal_copy)

        # Sort by similarity
        matches.sort(key=lambda x: x["pattern_similarity"], reverse=True)
        return matches[:top_k]

    def get_cohort_insights(
        self, industry: str, deal_size_bucket: str
    ) -> Dict:
        """
        Get aggregated insights for a deal cohort.
        Answers: "What does success look like for this type of deal?"
        """
        if not self.is_built:
            raise RuntimeError("Index not built.")

        cohort = [
            d
            for d in self.metadata
            if d.get("industry") == industry
            and d.get("deal_size_bucket") == deal_size_bucket
        ]

        if not cohort:
            return {"cohort_size": 0, "message": "No comparable deals found"}

        cohort_df = pd.DataFrame(cohort)
        won = cohort_df[cohort_df["outcome"] == "won"]
        lost = cohort_df[cohort_df["outcome"] == "lost"]

        insights = {
            "cohort_size": len(cohort),
            "win_rate": round(len(won) / len(cohort), 3) if cohort else 0,
            "avg_duration_won": round(won["duration_days"].mean(), 1) if not won.empty else None,
            "avg_duration_lost": round(lost["duration_days"].mean(), 1) if not lost.empty else None,
            "avg_engagement_won": round(won["engagement_per_week"].mean(), 3) if not won.empty else None,
            "avg_engagement_lost": round(lost["engagement_per_week"].mean(), 3) if not lost.empty else None,
            "avg_stakeholders_won": round(won["stakeholder_count"].mean(), 1) if not won.empty else None,
            "avg_stakeholders_lost": round(lost["stakeholder_count"].mean(), 1) if not lost.empty else None,
        }

        return insights

    def save(self, path: str = FAISS_INDEX_PATH) -> None:
        """Save index and metadata to disk."""
        os.makedirs(path, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))

        # Save metadata
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(self.metadata, f, default=str)

    def load(self, path: str = FAISS_INDEX_PATH) -> None:
        """Load index and metadata from disk."""
        index_path = os.path.join(path, "index.faiss")
        metadata_path = os.path.join(path, "metadata.json")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"No index found at {index_path}")

        self.index = faiss.read_index(index_path)

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.is_built = True
