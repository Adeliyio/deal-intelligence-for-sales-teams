"""
API dependencies — shared resources loaded once at startup.

Manages ML models, vector store, and feature data as singletons
so they're not reloaded on every request.
"""
import os
import pandas as pd
from typing import Optional

from backend.ml.win_model import WinProbabilityModel
from backend.ml.risk_model import RiskClassificationModel
from backend.ml.preprocessing import prepare_inference_data
from backend.rag.vector_store import DealVectorStore
from backend.rag.retriever import DealRetriever


class AppState:
    """Singleton holding loaded models and data."""

    def __init__(self):
        self.win_model: Optional[WinProbabilityModel] = None
        self.risk_model: Optional[RiskClassificationModel] = None
        self.retriever: Optional[DealRetriever] = None
        self.features_df: Optional[pd.DataFrame] = None
        self.is_ready = False

    def load(self):
        """Load all models and data at startup."""
        print("Loading models and data...")

        # Load feature data
        features_path = "data/processed/deal_features.csv"
        if os.path.exists(features_path):
            self.features_df = pd.read_csv(features_path)
            if "has_economic_buyer" in self.features_df.columns:
                self.features_df["has_economic_buyer"] = (
                    self.features_df["has_economic_buyer"].astype(int)
                )
            print(f"  Loaded {len(self.features_df)} deals from features")

        # Load win model
        win_model_path = "models/saved/win_model.joblib"
        if os.path.exists(win_model_path):
            self.win_model = WinProbabilityModel()
            self.win_model.load(win_model_path)
            print("  Win model loaded")

        # Load risk model
        risk_model_path = "models/saved/risk_model.joblib"
        if os.path.exists(risk_model_path):
            self.risk_model = RiskClassificationModel()
            self.risk_model.load(risk_model_path)
            print("  Risk model loaded")

        # Build vector store from features (needs normalization params)
        if self.features_df is not None:
            store = DealVectorStore(embedding_mode="feature")
            store.build_index(self.features_df)
            self.retriever = DealRetriever(store)
            print("  Vector store built")

        self.is_ready = True
        print("All resources loaded.")

    def get_deal_features(self, deal_id: str) -> Optional[dict]:
        """Look up features for a specific deal."""
        if self.features_df is None:
            return None

        deal = self.features_df[self.features_df["deal_id"] == deal_id]
        if deal.empty:
            return None

        return deal.iloc[0].to_dict()


# Global singleton
app_state = AppState()


def get_app_state() -> AppState:
    """Dependency injection for FastAPI routes."""
    return app_state
