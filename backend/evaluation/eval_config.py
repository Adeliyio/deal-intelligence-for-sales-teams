"""
Evaluation harness configuration.

Defines metrics, thresholds, and A/B test parameters for measuring
system performance — both ML model quality and agent impact.
"""

# Model evaluation metrics to compute
MODEL_METRICS = [
    "auc_roc",
    "brier_score",
    "calibration_curve",
    "precision_recall",
    "confusion_matrix",
    "feature_importance_shap",
]

# Calibration curve settings
CALIBRATION_N_BINS = 10
CALIBRATION_STRATEGY = "uniform"  # "uniform" or "quantile"

# SHAP analysis
SHAP_MAX_SAMPLES = 100  # Max samples for SHAP computation (expensive)
SHAP_PLOT_TOP_N = 15  # Top N features to display

# Critic Agent A/B test parameters
AB_TEST_N_DEALS = 200  # Number of simulated deals for A/B test
AB_TEST_SEED = 42
AB_TEST_METRICS = [
    "decision_quality",  # Agreement with ground truth action
    "false_urgency_rate",  # Unnecessary escalations
    "calibration_improvement",  # Risk score calibration delta
]

# False urgency definition
FALSE_URGENCY_THRESHOLD = 0.65  # Risk score above this triggers escalation
HEALTHY_DEAL_WIN_PROB = 0.55  # Deals above this are "actually healthy"

# Decision quality scoring
DECISION_ACTIONS = [
    "maintain_cadence",  # Keep current approach
    "increase_engagement",  # More touchpoints
    "escalate_executive",  # Bring in leadership
    "offer_concession",  # Discount or terms adjustment
    "deprioritize",  # Reduce effort on this deal
]

# Failure mode categories to document
FAILURE_MODES = [
    "sparse_activity",  # Fewer than 5 logged activities
    "long_cycle_enterprise",  # Enterprise deals with very long sales cycles
    "inconsistent_logging",  # Rep logging behavior is highly inconsistent
    "no_comparable_deals",  # First-time buyer profile with no history
]

# Minimum activities threshold for reliable prediction
MIN_ACTIVITIES_RELIABLE = 5

# Experiment tracking
EXPERIMENT_PROJECT = "deal-intelligence"
LOG_ARTIFACTS = True
