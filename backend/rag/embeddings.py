"""
Embedding generation for deal records.

Converts deal features and metadata into dense vector representations
for semantic similarity search.
"""
import numpy as np
import pandas as pd
from typing import List, Optional

from backend.rag.rag_config import (
    EMBEDDING_DIMENSION,
    DEAL_EMBED_FIELDS,
    FEATURE_EMBED_FIELDS,
)


class DealEmbedder:
    """
    Generates embeddings for deal records.

    Supports two modes:
    - API-based: uses OpenAI embeddings for semantic text matching
    - Feature-based: uses normalized feature vectors for numerical similarity

    The feature-based mode works without API keys and is used for
    deal-to-deal comparison based on quantitative signals.
    """

    def __init__(self, mode: str = "feature"):
        """
        Args:
            mode: "feature" for numerical similarity, "api" for OpenAI embeddings
        """
        self.mode = mode
        self._openai_client = None

    def _get_openai_client(self):
        """Lazy-load OpenAI client."""
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI()
        return self._openai_client

    def embed_deals(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Generate embeddings for all deals in the feature matrix.

        Returns:
            numpy array of shape (n_deals, embedding_dim)
        """
        if self.mode == "feature":
            return self._feature_embeddings(features_df)
        else:
            return self._api_embeddings(features_df)

    def embed_query(self, query_features: dict) -> np.ndarray:
        """
        Generate embedding for a single deal query.

        Args:
            query_features: dict of feature name -> value

        Returns:
            numpy array of shape (embedding_dim,)
        """
        if self.mode == "feature":
            return self._feature_embedding_single(query_features)
        else:
            return self._api_embedding_single(query_features)

    def _feature_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create normalized feature vectors for numerical similarity.

        Uses z-score normalization so all features contribute equally
        to distance calculations regardless of scale.
        """
        available_features = [
            f for f in FEATURE_EMBED_FIELDS if f in df.columns
        ]

        if not available_features:
            raise ValueError("No embedding features found in dataframe")

        # Extract and normalize
        feature_matrix = df[available_features].copy()

        # Handle booleans
        for col in feature_matrix.columns:
            if feature_matrix[col].dtype == bool:
                feature_matrix[col] = feature_matrix[col].astype(float)

        # Fill NaN with 0 (neutral after normalization)
        feature_matrix = feature_matrix.fillna(0)

        # Z-score normalization
        means = feature_matrix.mean()
        stds = feature_matrix.std()
        stds = stds.replace(0, 1)  # Avoid division by zero

        normalized = (feature_matrix - means) / stds

        # Store normalization params for query embedding
        self._norm_means = means
        self._norm_stds = stds
        self._feature_names = available_features

        return normalized.values

    def _feature_embedding_single(self, query_features: dict) -> np.ndarray:
        """Create normalized vector for a single deal query."""
        if not hasattr(self, "_norm_means"):
            raise RuntimeError(
                "Must call embed_deals() first to establish normalization parameters"
            )

        vector = []
        for feat in self._feature_names:
            val = query_features.get(feat, 0)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                val = 0
            # Apply same normalization
            normalized_val = (val - self._norm_means[feat]) / self._norm_stds[feat]
            vector.append(normalized_val)

        return np.array(vector)

    def _api_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """Generate OpenAI text embeddings from deal descriptions."""
        client = self._get_openai_client()
        texts = self._deals_to_text(df)

        embeddings = []
        # Batch API calls (max 2048 inputs per call)
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = client.embeddings.create(
                input=batch, model="text-embedding-3-small"
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)

        return np.array(embeddings)

    def _api_embedding_single(self, query_features: dict) -> np.ndarray:
        """Generate OpenAI embedding for a single query."""
        client = self._get_openai_client()
        text = self._deal_to_text(query_features)

        response = client.embeddings.create(
            input=[text], model="text-embedding-3-small"
        )
        return np.array(response.data[0].embedding)

    def _deals_to_text(self, df: pd.DataFrame) -> List[str]:
        """Convert deal records to text descriptions for embedding."""
        texts = []
        for _, row in df.iterrows():
            texts.append(self._deal_to_text(row.to_dict()))
        return texts

    def _deal_to_text(self, deal: dict) -> str:
        """Convert a single deal to a text description."""
        parts = []

        if "industry" in deal:
            parts.append(f"Industry: {deal['industry']}")
        if "deal_size_bucket" in deal:
            parts.append(f"Size: {deal['deal_size_bucket']}")
        if "deal_value" in deal:
            parts.append(f"Value: ${deal.get('deal_value', 0):,.0f}")
        if "duration_days" in deal:
            parts.append(f"Duration: {deal['duration_days']} days")
        if "engagement_per_week" in deal:
            parts.append(f"Engagement/week: {deal.get('engagement_per_week', 0):.2f}")
        if "silence_gap_days" in deal:
            parts.append(f"Silence: {deal.get('silence_gap_days', 0):.0f} days")
        if "stakeholder_count" in deal:
            parts.append(f"Stakeholders: {deal['stakeholder_count']}")
        if "outcome" in deal:
            parts.append(f"Outcome: {deal['outcome']}")

        return ". ".join(parts)
