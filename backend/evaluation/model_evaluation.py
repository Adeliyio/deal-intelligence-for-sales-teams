"""
Model evaluation harness.

Computes comprehensive metrics for the win probability and risk models:
- AUC-ROC with confidence intervals
- Brier score (calibration quality)
- Calibration curves (predicted vs actual probability)
- SHAP feature importance analysis
- Confusion matrix at operating threshold
- Performance by cohort (failure mode detection)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    brier_score_loss,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.calibration import calibration_curve
from typing import Dict, List, Tuple, Optional

from backend.evaluation.eval_config import (
    CALIBRATION_N_BINS,
    CALIBRATION_STRATEGY,
    SHAP_MAX_SAMPLES,
    SHAP_PLOT_TOP_N,
    MIN_ACTIVITIES_RELIABLE,
    FAILURE_MODES,
)


class ModelEvaluator:
    """
    Comprehensive evaluation of ML model performance.

    Goes beyond simple accuracy to measure what actually matters:
    - Are the probabilities calibrated? (Brier score, calibration curve)
    - Where does the model fail? (cohort-level breakdown)
    - What drives predictions? (SHAP values, not just feature importance)
    """

    def __init__(self, model, model_name: str = "model"):
        self.model = model
        self.model_name = model_name
        self.results = {}

    def full_evaluation(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        features_df: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Run the complete evaluation suite.

        Args:
            X_test: Test features
            y_test: True labels
            features_df: Full feature DataFrame (for cohort analysis)
        """
        # Get predictions
        if hasattr(self.model, "calibrated_model") and self.model.calibrated_model:
            y_proba = self.model.calibrated_model.predict_proba(X_test)[:, 1]
            y_proba_raw = self.model.base_model.predict_proba(X_test)[:, 1]
        else:
            y_proba = self.model.predict_proba(X_test)[:, 1]
            y_proba_raw = y_proba

        self.results = {
            "model_name": self.model_name,
            "n_test_samples": len(y_test),
            "discrimination": self._evaluate_discrimination(y_test, y_proba),
            "calibration": self._evaluate_calibration(y_test, y_proba, y_proba_raw),
            "threshold_analysis": self._evaluate_thresholds(y_test, y_proba),
            "shap_analysis": self._shap_analysis(X_test),
        }

        # Failure mode analysis if full features available
        if features_df is not None:
            self.results["failure_modes"] = self._failure_mode_analysis(
                features_df, X_test, y_test, y_proba
            )

        return self.results

    def _evaluate_discrimination(
        self, y_test: pd.Series, y_proba: np.ndarray
    ) -> Dict:
        """AUC-ROC with bootstrap confidence interval."""
        auc = roc_auc_score(y_test, y_proba)

        # Bootstrap CI
        n_bootstrap = 1000
        rng = np.random.RandomState(42)
        aucs = []
        for _ in range(n_bootstrap):
            indices = rng.randint(0, len(y_test), len(y_test))
            if len(np.unique(y_test.iloc[indices])) < 2:
                continue
            aucs.append(roc_auc_score(y_test.iloc[indices], y_proba[indices]))

        ci_lower = np.percentile(aucs, 2.5) if aucs else auc
        ci_upper = np.percentile(aucs, 97.5) if aucs else auc

        # ROC curve points
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)

        return {
            "auc_roc": round(auc, 4),
            "auc_ci_lower": round(ci_lower, 4),
            "auc_ci_upper": round(ci_upper, 4),
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
            },
        }

    def _evaluate_calibration(
        self,
        y_test: pd.Series,
        y_proba_calibrated: np.ndarray,
        y_proba_raw: np.ndarray,
    ) -> Dict:
        """
        Calibration analysis — the most important metric for this system.

        A model that says "82% likely to close" must be right 82% of the time
        for that claim to be meaningful. Brier score and calibration curves
        measure this directly.
        """
        brier_calibrated = brier_score_loss(y_test, y_proba_calibrated)
        brier_raw = brier_score_loss(y_test, y_proba_raw)

        # Calibration curve
        prob_true, prob_pred = calibration_curve(
            y_test,
            y_proba_calibrated,
            n_bins=CALIBRATION_N_BINS,
            strategy=CALIBRATION_STRATEGY,
        )

        # Raw calibration curve for comparison
        prob_true_raw, prob_pred_raw = calibration_curve(
            y_test,
            y_proba_raw,
            n_bins=CALIBRATION_N_BINS,
            strategy=CALIBRATION_STRATEGY,
        )

        # Expected Calibration Error (ECE)
        ece = self._compute_ece(y_test, y_proba_calibrated)
        ece_raw = self._compute_ece(y_test, y_proba_raw)

        return {
            "brier_score_calibrated": round(brier_calibrated, 4),
            "brier_score_raw": round(brier_raw, 4),
            "brier_improvement": round(brier_raw - brier_calibrated, 4),
            "ece_calibrated": round(ece, 4),
            "ece_raw": round(ece_raw, 4),
            "calibration_curve": {
                "prob_true": prob_true.tolist(),
                "prob_pred": prob_pred.tolist(),
            },
            "calibration_curve_raw": {
                "prob_true": prob_true_raw.tolist(),
                "prob_pred": prob_pred_raw.tolist(),
            },
        }

    def _compute_ece(self, y_true: pd.Series, y_proba: np.ndarray) -> float:
        """Expected Calibration Error ��� weighted average of bin calibration errors."""
        bin_edges = np.linspace(0, 1, CALIBRATION_N_BINS + 1)
        ece = 0.0

        for i in range(CALIBRATION_N_BINS):
            mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            bin_acc = y_true[mask].mean()
            bin_conf = y_proba[mask].mean()
            bin_weight = mask.sum() / len(y_true)
            ece += bin_weight * abs(bin_acc - bin_conf)

        return ece

    def _evaluate_thresholds(
        self, y_test: pd.Series, y_proba: np.ndarray
    ) -> Dict:
        """Analyze model performance at different operating thresholds."""
        thresholds_to_test = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]
        results = []

        for thresh in thresholds_to_test:
            y_pred = (y_proba >= thresh).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            results.append({
                "threshold": thresh,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
            })

        # Find optimal threshold (max F1)
        best = max(results, key=lambda x: x["f1"])

        return {
            "threshold_analysis": results,
            "optimal_threshold": best["threshold"],
            "optimal_f1": best["f1"],
        }

    def _shap_analysis(self, X_test: pd.DataFrame) -> Dict:
        """SHAP feature importance — what drives predictions and what is noise."""
        try:
            import shap

            # Limit samples for performance
            n_samples = min(SHAP_MAX_SAMPLES, len(X_test))
            X_sample = X_test.iloc[:n_samples]

            explainer = shap.TreeExplainer(self.model.base_model)
            shap_values = explainer.shap_values(X_sample)

            # Mean absolute SHAP values per feature
            mean_shap = np.abs(shap_values).mean(axis=0)
            feature_importance = pd.DataFrame({
                "feature": X_test.columns.tolist(),
                "mean_shap_value": mean_shap,
            }).sort_values("mean_shap_value", ascending=False)

            return {
                "top_features": feature_importance.head(SHAP_PLOT_TOP_N).to_dict("records"),
                "n_samples_analyzed": n_samples,
            }

        except Exception as e:
            return {"error": str(e), "fallback": "Using built-in feature importance"}

    def _failure_mode_analysis(
        self,
        features_df: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_proba: np.ndarray,
    ) -> Dict:
        """
        Document where the model performs poorly.

        Naming what breaks builds more credibility than only showing what works.
        """
        # Align indices
        test_indices = X_test.index
        test_features = features_df.loc[test_indices] if test_indices[0] in features_df.index else features_df.iloc[:len(X_test)]

        failure_analysis = {}

        # 1. Sparse activity deals
        sparse_mask = test_features["total_activities"] < MIN_ACTIVITIES_RELIABLE
        if sparse_mask.sum() > 0:
            sparse_proba = y_proba[sparse_mask.values]
            sparse_true = y_test[sparse_mask.values]
            failure_analysis["sparse_activity"] = {
                "description": f"Deals with fewer than {MIN_ACTIVITIES_RELIABLE} logged activities",
                "n_deals": int(sparse_mask.sum()),
                "avg_predicted_proba": round(float(sparse_proba.mean()), 4),
                "actual_win_rate": round(float(sparse_true.mean()), 4),
                "calibration_error": round(
                    abs(float(sparse_proba.mean()) - float(sparse_true.mean())), 4
                ),
            }

        # 2. Enterprise long-cycle deals
        if "deal_size_bucket" in test_features.columns:
            enterprise_mask = test_features["deal_size_bucket"] == "enterprise"
            if enterprise_mask.sum() > 0:
                ent_proba = y_proba[enterprise_mask.values]
                ent_true = y_test[enterprise_mask.values]
                failure_analysis["long_cycle_enterprise"] = {
                    "description": "Enterprise deals with long sales cycles",
                    "n_deals": int(enterprise_mask.sum()),
                    "avg_predicted_proba": round(float(ent_proba.mean()), 4),
                    "actual_win_rate": round(float(ent_true.mean()), 4),
                    "calibration_error": round(
                        abs(float(ent_proba.mean()) - float(ent_true.mean())), 4
                    ),
                }

        # 3. High null rate (inconsistent logging proxy)
        null_features = X_test.isnull().sum(axis=1)
        high_null_mask = null_features > (X_test.shape[1] * 0.3)
        if high_null_mask.sum() > 0:
            hn_proba = y_proba[high_null_mask.values]
            hn_true = y_test[high_null_mask.values]
            failure_analysis["inconsistent_logging"] = {
                "description": "Deals with >30% missing feature values",
                "n_deals": int(high_null_mask.sum()),
                "avg_predicted_proba": round(float(hn_proba.mean()), 4),
                "actual_win_rate": round(float(hn_true.mean()), 4),
            }

        # 4. Overall error analysis
        errors = np.abs(y_proba - y_test.values)
        worst_indices = np.argsort(errors)[-5:]  # Top 5 worst predictions
        failure_analysis["worst_predictions"] = {
            "description": "Deals with largest prediction error",
            "deals": [
                {
                    "index": int(idx),
                    "predicted": round(float(y_proba[idx]), 4),
                    "actual": int(y_test.iloc[idx]),
                    "error": round(float(errors[idx]), 4),
                }
                for idx in worst_indices
            ],
        }

        return failure_analysis

    def summary_report(self) -> str:
        """Generate human-readable evaluation summary."""
        if not self.results:
            return "No evaluation results. Run full_evaluation() first."

        lines = [
            f"{'='*60}",
            f"EVALUATION REPORT: {self.model_name}",
            f"{'='*60}",
            f"Test samples: {self.results['n_test_samples']}",
            "",
            "--- Discrimination ---",
            f"AUC-ROC: {self.results['discrimination']['auc_roc']} "
            f"[{self.results['discrimination']['auc_ci_lower']}, "
            f"{self.results['discrimination']['auc_ci_upper']}]",
            "",
            "--- Calibration ---",
            f"Brier Score (calibrated): {self.results['calibration']['brier_score_calibrated']}",
            f"Brier Score (raw): {self.results['calibration']['brier_score_raw']}",
            f"Calibration improvement: {self.results['calibration']['brier_improvement']}",
            f"ECE (calibrated): {self.results['calibration']['ece_calibrated']}",
            "",
            "--- Optimal Threshold ---",
            f"Threshold: {self.results['threshold_analysis']['optimal_threshold']}",
            f"F1 at optimal: {self.results['threshold_analysis']['optimal_f1']}",
        ]

        if "shap_analysis" in self.results:
            shap = self.results["shap_analysis"]
            if "top_features" in shap:
                lines.extend(["", "--- Top SHAP Features ---"])
                for item in shap["top_features"][:5]:
                    lines.append(
                        f"  {item['feature']}: {item['mean_shap_value']:.4f}"
                    )

        if "failure_modes" in self.results:
            lines.extend(["", "--- Failure Modes ---"])
            for mode, info in self.results["failure_modes"].items():
                if mode == "worst_predictions":
                    continue
                lines.append(f"  {mode}: {info.get('description', '')}")
                lines.append(f"    N={info.get('n_deals', 0)}, "
                           f"Cal Error={info.get('calibration_error', 'N/A')}")

        lines.append(f"\n{'='*60}")
        return "\n".join(lines)
