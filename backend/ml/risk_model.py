"""
Risk classification model.

Binary classifier: will this deal go cold in the next 14 days?

Design decision: Binary classification over survival analysis.
Survival models assume the event is inevitable and model *when* it occurs.
For deal risk, we want to model *whether* a deal stalls, not when it closes.
"""
import numpy as np
import pandas as pd
import joblib
import os
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from typing import Dict, Tuple

from backend.ml.model_config import (
    RISK_MODEL_PARAMS,
    CALIBRATION_METHOD,
    RISK_THRESHOLD,
)


class RiskClassificationModel:
    """
    Predicts whether a deal will stall within the next 14 days.

    The model focuses on temporal signals — engagement decay, silence gaps,
    and velocity changes — since these are the strongest early indicators
    of deal stagnation.
    """

    def __init__(self):
        self.base_model = XGBClassifier(**RISK_MODEL_PARAMS)
        self.calibrated_model = None
        self.is_fitted = False
        self.training_stats = {}

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict:
        """Train risk model with calibration."""
        # Adjust for class imbalance
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        if n_pos > 0:
            self.base_model.set_params(scale_pos_weight=n_neg / n_pos)

        # Train
        self.base_model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Calibrate
        n_samples = len(X_train)
        self.calibrated_model = CalibratedClassifierCV(
            self.base_model,
            method=CALIBRATION_METHOD,
            cv=min(5, max(2, n_samples // 10)),
        )
        self.calibrated_model.fit(X_train, y_train)

        self.is_fitted = True
        self.training_stats = {
            "n_train": n_samples,
            "n_test": len(X_test),
            "class_balance": {"at_risk": int(n_pos), "healthy": int(n_neg)},
        }

        metrics = self.evaluate(X_test, y_test)
        self.training_stats["metrics"] = metrics

        return metrics

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate on held-out test set."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before evaluation.")

        cal_proba = self.calibrated_model.predict_proba(X_test)[:, 1]
        y_pred = (cal_proba >= RISK_THRESHOLD).astype(int)

        # Precision-recall curve to find optimal threshold
        precision, recall, thresholds = precision_recall_curve(y_test, cal_proba)

        # F1 at each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_f1_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else RISK_THRESHOLD

        metrics = {
            "auc_roc": round(roc_auc_score(y_test, cal_proba), 4),
            "brier_score": round(brier_score_loss(y_test, cal_proba), 4),
            "operating_threshold": RISK_THRESHOLD,
            "optimal_threshold": round(float(optimal_threshold), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True
            ),
        }

        return metrics

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict deal risk scores.

        Returns DataFrame with:
        - risk_score: calibrated probability of stalling
        - risk_level: categorical (low / medium / high / critical)
        - is_at_risk: binary flag at operating threshold
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before prediction.")

        cal_proba = self.calibrated_model.predict_proba(X)[:, 1]

        results = pd.DataFrame(
            {
                "risk_score": np.round(cal_proba, 4),
                "is_at_risk": (cal_proba >= RISK_THRESHOLD).astype(int),
                "risk_level": pd.cut(
                    cal_proba,
                    bins=[0, 0.25, 0.50, 0.75, 1.0],
                    labels=["low", "medium", "high", "critical"],
                ),
            }
        )

        return results

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """Feature importance ranked by contribution to risk prediction."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained first.")

        importance = self.base_model.feature_importances_
        fi_df = pd.DataFrame(
            {"feature": feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        fi_df["importance_pct"] = (
            fi_df["importance"] / fi_df["importance"].sum() * 100
        ).round(2)

        return fi_df

    def save(self, path: str = "models/saved/risk_model.joblib") -> None:
        """Save trained model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(
            {
                "base_model": self.base_model,
                "calibrated_model": self.calibrated_model,
                "training_stats": self.training_stats,
            },
            path,
        )

    def load(self, path: str = "models/saved/risk_model.joblib") -> None:
        """Load trained model from disk."""
        data = joblib.load(path)
        self.base_model = data["base_model"]
        self.calibrated_model = data["calibrated_model"]
        self.training_stats = data["training_stats"]
        self.is_fitted = True
