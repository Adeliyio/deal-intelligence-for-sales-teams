"""
Model training pipeline.

Trains the win probability model and risk classification model,
outputs evaluation metrics, and saves trained models.

Run from project root: python -m scripts.train_models
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.ml.preprocessing import (
    load_and_prepare_data,
    prepare_win_model_data,
    prepare_risk_model_data,
)
from backend.ml.win_model import WinProbabilityModel
from backend.ml.risk_model import RiskClassificationModel
from backend.ml.model_config import WIN_MODEL_FEATURES, RISK_MODEL_FEATURES


def train_win_model(df):
    """Train and evaluate the win probability model."""
    print("\n" + "=" * 60)
    print("WIN PROBABILITY MODEL")
    print("=" * 60)

    X_train, X_test, y_train, y_test = prepare_win_model_data(df)
    print(f"\nTraining set: {len(X_train)} deals")
    print(f"Test set: {len(X_test)} deals")
    print(f"Win rate (train): {y_train.mean():.1%}")

    model = WinProbabilityModel()
    metrics = model.train(X_train, y_train, X_test, y_test)

    print(f"\n--- Evaluation Metrics ---")
    print(f"AUC-ROC: {metrics['auc_roc']}")
    print(f"Brier Score (raw): {metrics['brier_score_raw']}")
    print(f"Brier Score (calibrated): {metrics['brier_score_calibrated']}")
    print(f"Operating threshold: {metrics['operating_threshold']}")
    print(f"\nConfusion Matrix (at threshold {metrics['operating_threshold']}):")
    cm = metrics["confusion_matrix"]
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")

    # Feature importance
    print(f"\n--- Feature Importance (Top 10) ---")
    fi = model.get_feature_importance(WIN_MODEL_FEATURES)
    print(fi.head(10).to_string(index=False))

    # Save model
    model.save("models/saved/win_model.joblib")
    print(f"\nModel saved to models/saved/win_model.joblib")

    return model, metrics


def train_risk_model(df):
    """Train and evaluate the risk classification model."""
    print("\n" + "=" * 60)
    print("RISK CLASSIFICATION MODEL")
    print("=" * 60)

    X_train, X_test, y_train, y_test = prepare_risk_model_data(df)
    print(f"\nTraining set: {len(X_train)} deals")
    print(f"Test set: {len(X_test)} deals")
    print(f"At-risk rate (train): {y_train.mean():.1%}")

    model = RiskClassificationModel()
    metrics = model.train(X_train, y_train, X_test, y_test)

    print(f"\n--- Evaluation Metrics ---")
    print(f"AUC-ROC: {metrics['auc_roc']}")
    print(f"Brier Score: {metrics['brier_score']}")
    print(f"Operating threshold: {metrics['operating_threshold']}")
    print(f"Optimal threshold (max F1): {metrics['optimal_threshold']}")
    print(f"\nConfusion Matrix (at threshold {metrics['operating_threshold']}):")
    cm = metrics["confusion_matrix"]
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")

    # Feature importance
    print(f"\n--- Feature Importance ---")
    fi = model.get_feature_importance(RISK_MODEL_FEATURES)
    print(fi.to_string(index=False))

    # Save model
    model.save("models/saved/risk_model.joblib")
    print(f"\nModel saved to models/saved/risk_model.joblib")

    return model, metrics


def main():
    print("=" * 60)
    print("Deal Intelligence — Model Training Pipeline")
    print("=" * 60)

    # Load data
    print("\nLoading feature matrix...")
    df = load_and_prepare_data()
    print(f"Loaded {len(df)} deals with {df.shape[1]} features")

    # Train models
    win_model, win_metrics = train_win_model(df)
    risk_model, risk_metrics = train_risk_model(df)

    # Save metrics summary
    os.makedirs("models/saved", exist_ok=True)
    summary = {
        "win_model": win_metrics,
        "risk_model": risk_metrics,
    }
    with open("models/saved/training_metrics.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("Training complete. Metrics saved to models/saved/training_metrics.json")
    print("=" * 60)

    # Demo: predict on test data
    print("\n--- Demo: Win Probability Predictions (first 5 deals) ---")
    from backend.ml.preprocessing import prepare_inference_data

    sample = df.head(5)
    X_sample = prepare_inference_data(sample, model_type="win")
    predictions = win_model.predict(X_sample)
    demo = sample[["deal_id", "outcome", "deal_value"]].reset_index(drop=True)
    demo = demo.join(predictions)
    print(demo.to_string(index=False))


if __name__ == "__main__":
    main()
