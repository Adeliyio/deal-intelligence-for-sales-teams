"""
Centralized configuration for feature engineering constants and weights.
All tunable parameters live here to avoid magic numbers scattered in code.
"""

# Deal stages in order of progression
STAGE_ORDER = [
    "prospecting",
    "qualification",
    "proposal",
    "negotiation",
    "closed_won",
    "closed_lost",
]

# Expected days per stage (median) — used for synthetic data generation
STAGE_DURATION_MEDIAN = {
    "prospecting": 7,
    "qualification": 10,
    "proposal": 12,
    "negotiation": 14,
}

# Activity types and their engagement weights
ENGAGEMENT_WEIGHTS = {
    "email_opened": 0.1,
    "email_received": 0.3,
    "meeting": 0.4,
    "demo": 0.2,
    "call_inbound": 0.25,
}

# Activity types considered "meaningful" for silence gap computation
# (only inbound / two-way activities count)
MEANINGFUL_ACTIVITY_TYPES = [
    "email_received",
    "meeting",
    "call_inbound",
    "demo",
]

# Decay-weighted engagement
DECAY_LAMBDA = 0.05  # Half-life ~14 days: ln(2)/0.05 ≈ 13.86 days

# Silence gap severity thresholds
SILENCE_BASELINE_WEEKS = 3  # Weeks of deal history used to compute baseline frequency
SILENCE_SEVERE_THRESHOLD = 3.0  # Severity score above this is a strong churn signal

# Cohort fields for deal velocity comparison
COHORT_FIELDS = ["deal_size_bucket", "industry"]
MIN_COHORT_SIZE = 5  # Fall back to broader cohort if fewer members

# Deal size buckets and their value ranges
DEAL_SIZE_RANGES = {
    "smb": (5_000, 25_000),
    "mid_market": (25_000, 100_000),
    "enterprise": (100_000, 500_000),
}

# Industries for synthetic data
INDUSTRIES = [
    "technology",
    "healthcare",
    "financial_services",
    "manufacturing",
    "retail",
    "education",
    "media",
]

# Lead sources
LEAD_SOURCES = ["inbound", "outbound", "referral"]

# Contact roles
CONTACT_ROLES = [
    "champion",
    "economic_buyer",
    "technical_evaluator",
    "end_user",
    "blocker",
]

# Contact seniority levels
SENIORITY_LEVELS = ["c_level", "vp", "director", "manager", "individual_contributor"]

# Rep names for synthetic data
REP_NAMES = [
    "Sarah Chen",
    "Marcus Johnson",
    "Elena Rodriguez",
    "David Park",
    "Aisha Patel",
]

# Response time matching window (days) — max gap between outbound and inbound
# for them to be considered a request-reply pair
RESPONSE_TIME_WINDOW_DAYS = 14
