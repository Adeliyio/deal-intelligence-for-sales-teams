"""
ML model configuration — hyperparameters, thresholds, and training settings.
"""

# XGBoost win probability model hyperparameters
WIN_MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1.0,  # Adjusted during training based on class balance
    "random_state": 42,
    "eval_metric": "logloss",
}

# Risk model hyperparameters (binary: will deal stall in 14 days?)
RISK_MODEL_PARAMS = {
    "n_estimators": 150,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "eval_metric": "logloss",
}

# Training configuration
TRAIN_TEST_SPLIT = 0.20  # 20% held-out test set
STRATIFY_COLUMN = "outcome"
RANDOM_STATE = 42

# Platt scaling (calibration)
CALIBRATION_METHOD = "sigmoid"  # sigmoid = Platt scaling, isotonic = isotonic regression
CALIBRATION_CV = 5

# Features used by the win probability model
WIN_MODEL_FEATURES = [
    "avg_response_time_hours",
    "median_response_time_hours",
    "response_count",
    "engagement_score",
    "engagement_per_week",
    "total_activities",
    "avg_days_per_stage",
    "total_transitions",
    "silence_gap_days",
    "silence_gap_severity",
    "stakeholder_count",
    "has_economic_buyer",
    "decision_maker_ratio",
    "stakeholder_breadth",
    "decay_weighted_engagement",
    "deal_value",
    "duration_days",
    "deal_velocity_ratio",
]

# Features used by the risk model (subset — focuses on temporal signals)
RISK_MODEL_FEATURES = [
    "avg_response_time_hours",
    "engagement_per_week",
    "silence_gap_days",
    "silence_gap_severity",
    "decay_weighted_engagement",
    "deal_velocity_ratio",
    "total_activities",
    "duration_days",
    "stakeholder_count",
    "decision_maker_ratio",
]

# Operating threshold for win probability (not default 0.5)
WIN_PROBABILITY_THRESHOLD = 0.45

# Risk classification threshold
RISK_THRESHOLD = 0.50

# Stagnation definition: days of silence to be labeled "at risk"
STAGNATION_WINDOW_DAYS = 14

# Confidence interval settings for cold-start
CONFIDENCE_INTERVAL_PERCENTILE = 0.90  # 90% CI
MIN_TRAINING_SAMPLES = 30  # Below this, output wide CI with warning
