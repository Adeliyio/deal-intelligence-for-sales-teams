"""
Win probability model.

XGBoost classifier with Platt scaling calibration.
Raw XGBoost scores are not probabilities — an uncalibrated 0.82 is not
"82% likely to close." Platt scaling maps outputs to true probabilities.
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
)
from typing import Dict, Tuple, Optional

from backend.ml.model_config import (
    WIN_MODEL_PARAMS,
    CALIBRATION_METHOD,
    CALIBRATION_CV,
    WIN_PROBABILITY_THRESHOLD,
    MIN_TRAINING_SAMPLES,
    CONFIDENCE_INTERVAL_PERCENTILE,
)


class WinProbabilityModel:
    """
    Calibrated XGBoost model for deal win probability prediction.

    Key design decisions:
    - XGBoost: handles missing values, tabular data, sparse CRM records
    - Platt scaling: converts raw scores to calibrated probabilities
    - Confidence intervals: explicit uncertainty for cold-start scenarios
    """

    def __init__(self):
        self.base_model = XGBClassifier(**WIN_MODEL_PARAMS)
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
        """
        Train the model with calibration.

        Returns a dictionary of evaluation metrics.
        """
        n_samples = len(X_train)

        # Adjust for class imbalance
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        if n_pos > 0:
            self.base_model.set_params(scale_pos_weight=n_neg / n_pos)

        # Train base XGBoost model
        self.base_model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Apply Platt scaling calibration
        self.calibrated_model = CalibratedClassifierCV(
            self.base_model,
            method=CALIBRATION_METHOD,
            cv=CALIBRATION_CV if n_samples >= 50 else 3,
        )
        self.calibrated_model.fit(X_train, y_train)

        self.is_fitted = True
        self.training_stats = {
            "n_train": n_samples,
            "n_test": len(X_test),
            "class_balance": {
                "won": int(n_pos),
                "lost": int(n_neg),
            },
        }

        # Evaluate
        metrics = self.evaluate(X_test, y_test)
        self.training_stats["metrics"] = metrics

        return metrics

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Compute evaluation metrics on held-out test set."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before evaluation.")

        # Raw (uncalibrated) predictions
        raw_proba = self.base_model.predict_proba(X_test)[:, 1]

        # Calibrated predictions
        cal_proba = self.calibrated_model.predict_proba(X_test)[:, 1]

        # Binary predictions at operating threshold
        y_pred = (cal_proba >= WIN_PROBABILITY_THRESHOLD).astype(int)

        # Metrics
        metrics = {
            "auc_roc": round(roc_auc_score(y_test, cal_proba), 4),
            "brier_score_raw": round(brier_score_loss(y_test, raw_proba), 4),
            "brier_score_calibrated": round(
                brier_score_loss(y_test, cal_proba), 4
            ),
            "operating_threshold": WIN_PROBABILITY_THRESHOLD,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True
            ),
        }

        return metrics

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict calibrated win probabilities with confidence intervals.

        Returns DataFrame with columns:
        - win_probability: calibrated score
        - confidence_lower: lower bound of CI
        - confidence_upper: upper bound of CI
        - confidence_width: width of CI (wider = less certain)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before prediction.")

        cal_proba = self.calibrated_model.predict_proba(X)[:, 1]

        # Compute confidence intervals via calibrator ensemble variance
        ci_lower, ci_upper = self._compute_confidence_intervals(X)

        results = pd.DataFrame(
            {
                "win_probability": np.round(cal_proba, 4),
                "confidence_lower": np.round(ci_lower, 4),
                "confidence_upper": np.round(ci_upper, 4),
                "confidence_width": np.round(ci_upper - ci_lower, 4),
            }
        )

        # Flag low-confidence predictions (cold-start awareness)
        n_train = self.training_stats.get("n_train", 0)
        if n_train < MIN_TRAINING_SAMPLES:
            results["low_confidence_warning"] = True
        else:
            results["low_confidence_warning"] = (
                results["confidence_width"] > 0.35
            )

        return results

    def _compute_confidence_intervals(
        self, X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate prediction intervals using the calibrated model's
        ensemble of calibrators (one per CV fold).
        """
        # Get predictions from each calibrator in the ensemble
        predictions = []
        for calibrator in self.calibrated_model.calibrated_classifiers_:
            proba = calibrator.predict_proba(X)[:, 1]
            predictions.append(proba)

        predictions = np.array(predictions)

        # Compute percentile-based confidence intervals
        alpha = (1 - CONFIDENCE_INTERVAL_PERCENTILE) / 2
        lower = np.percentile(predictions, alpha * 100, axis=0)
        upper = np.percentile(predictions, (1 - alpha) * 100, axis=0)

        # Clip to valid probability range
        lower = np.clip(lower, 0, 1)
        upper = np.clip(upper, 0, 1)

        return lower, upper

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """Get feature importance from the base XGBoost model."""
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

    def save(self, path: str = "models/saved/win_model.joblib") -> None:
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

    def load(self, path: str = "models/saved/win_model.joblib") -> None:
        """Load trained model from disk."""
        data = joblib.load(path)
        self.base_model = data["base_model"]
        self.calibrated_model = data["calibrated_model"]
        self.training_stats = data["training_stats"]
        self.is_fitted = True
