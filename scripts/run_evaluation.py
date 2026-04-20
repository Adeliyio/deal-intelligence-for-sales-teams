"""
Run the full evaluation harness.

Produces:
1. Model evaluation reports (AUC, calibration, SHAP, failure modes)
2. Critic Agent A/B test results
3. Saves all metrics to JSON for experiment tracking

Run from project root: python -m scripts.run_evaluation
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
from backend.evaluation.model_evaluation import ModelEvaluator
from backend.evaluation.critic_ab_test import CriticABTest


def evaluate_win_model(df):
    """Evaluate the win probability model."""
    print("\n[1/3] Evaluating Win Probability Model...")

    X_train, X_test, y_train, y_test = prepare_win_model_data(df)

    # Load trained model
    model = WinProbabilityModel()
    model_path = "models/saved/win_model.joblib"
    if not os.path.exists(model_path):
        print("  Model not found — training fresh...")
        model.train(X_train, y_train, X_test, y_test)
    else:
        model.load(model_path)

    # Run evaluation
    evaluator = ModelEvaluator(model, model_name="Win Probability Model")
    results = evaluator.full_evaluation(X_test, y_test, features_df=df)

    print(evaluator.summary_report())
    return results


def evaluate_risk_model(df):
    """Evaluate the risk classification model."""
    print("\n[2/3] Evaluating Risk Classification Model...")

    X_train, X_test, y_train, y_test = prepare_risk_model_data(df)

    # Load trained model
    model = RiskClassificationModel()
    model_path = "models/saved/risk_model.joblib"
    if not os.path.exists(model_path):
        print("  Model not found — training fresh...")
        model.train(X_train, y_train, X_test, y_test)
    else:
        model.load(model_path)

    # Run evaluation
    evaluator = ModelEvaluator(model, model_name="Risk Classification Model")
    results = evaluator.full_evaluation(X_test, y_test, features_df=df)

    print(evaluator.summary_report())
    return results


def run_critic_ab_test():
    """Run the Critic Agent A/B test."""
    print("\n[3/3] Running Critic Agent A/B Test...")

    ab_test = CriticABTest()
    results = ab_test.run_test()

    print(ab_test.summary_report())
    return results


def main():
    print("=" * 60)
    print("Deal Intelligence — Evaluation Harness")
    print("=" * 60)

    # Load data
    df = load_and_prepare_data()
    print(f"Loaded {len(df)} deals")

    # Run evaluations
    win_results = evaluate_win_model(df)
    risk_results = evaluate_risk_model(df)
    ab_results = run_critic_ab_test()

    # Save all results
    output_dir = "models/saved"
    os.makedirs(output_dir, exist_ok=True)

    all_results = {
        "win_model_evaluation": win_results,
        "risk_model_evaluation": risk_results,
        "critic_ab_test": ab_results,
    }

    output_path = os.path.join(output_dir, "evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n\nAll evaluation results saved to {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
