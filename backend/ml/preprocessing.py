"""
Data preprocessing for ML models.

Handles missing value imputation, feature encoding, and train/test splitting
with stratification by outcome.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

from backend.ml.model_config import (
    WIN_MODEL_FEATURES,
    RISK_MODEL_FEATURES,
    TRAIN_TEST_SPLIT,
    RANDOM_STATE,
    STAGNATION_WINDOW_DAYS,
)


def load_and_prepare_data(
    features_path: str = "data/processed/deal_features.csv",
) -> pd.DataFrame:
    """Load feature matrix and prepare for modeling."""
    df = pd.read_csv(features_path)

    # Convert boolean columns
    if "has_economic_buyer" in df.columns:
        df["has_economic_buyer"] = df["has_economic_buyer"].astype(int)

    return df


def create_win_labels(df: pd.DataFrame) -> pd.Series:
    """
    Binary labels for win probability model.
    1 = won, 0 = lost. Active deals are excluded from training.
    """
    return (df["outcome"] == "won").astype(int)


def create_risk_labels(df: pd.DataFrame) -> pd.Series:
    """
    Binary labels for risk model.
    1 = deal went cold (silence_gap_days > STAGNATION_WINDOW_DAYS or outcome=lost)
    0 = deal remained active/healthy
    """
    is_stagnant = df["silence_gap_days"].fillna(0) > STAGNATION_WINDOW_DAYS
    is_lost = df["outcome"] == "lost"
    return (is_stagnant | is_lost).astype(int)


def impute_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Handle missing values in feature columns.

    Strategy: median imputation for numeric features.
    XGBoost handles NaN natively, but we impute for calibration
    (sklearn's CalibratedClassifierCV does not support NaN).
    """
    df_imputed = df[features].copy()

    for col in features:
        if df_imputed[col].isnull().any():
            median_val = df_imputed[col].median()
            df_imputed[col] = df_imputed[col].fillna(median_val)

    return df_imputed


def prepare_win_model_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare train/test split for win probability model.
    Only uses closed deals (won/lost) — active deals excluded.
    """
    # Filter to closed deals only
    closed = df[df["outcome"].isin(["won", "lost"])].copy()

    X = impute_features(closed, WIN_MODEL_FEATURES)
    y = create_win_labels(closed)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TRAIN_TEST_SPLIT, stratify=y, random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test


def prepare_risk_model_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare train/test split for risk classification model.
    Uses all deals (including active ones).
    """
    X = impute_features(df, RISK_MODEL_FEATURES)
    y = create_risk_labels(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TRAIN_TEST_SPLIT, stratify=y, random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test


def prepare_inference_data(
    df: pd.DataFrame, model_type: str = "win"
) -> pd.DataFrame:
    """Prepare feature matrix for inference (no labels needed)."""
    features = WIN_MODEL_FEATURES if model_type == "win" else RISK_MODEL_FEATURES
    return impute_features(df, features)
