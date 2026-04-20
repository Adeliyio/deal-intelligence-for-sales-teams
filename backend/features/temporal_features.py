"""
Temporal feature engineering for deal intelligence.

Computes six core signals from raw CRM data:
1. Response time — proxy for buyer engagement
2. Engagement score — composite interest signal
3. Deal velocity — relative momentum vs cohort
4. Silence gap — stagnation detection with contextual severity
5. Stakeholder coverage — multi-threading signal
6. Decay-weighted engagement — recency-adjusted activity score
"""
import numpy as np
import pandas as pd
from typing import Optional

from backend.features.feature_config import (
    ENGAGEMENT_WEIGHTS,
    MEANINGFUL_ACTIVITY_TYPES,
    DECAY_LAMBDA,
    SILENCE_BASELINE_WEEKS,
    COHORT_FIELDS,
    MIN_COHORT_SIZE,
    RESPONSE_TIME_WINDOW_DAYS,
    STAGE_ORDER,
)


class TemporalFeatureEngineer:
    """Computes temporal features from raw CRM data."""

    def __init__(
        self,
        deals_df: pd.DataFrame,
        activities_df: pd.DataFrame,
        contacts_df: pd.DataFrame,
        stage_transitions_df: pd.DataFrame,
        reference_date: Optional[str] = None,
    ):
        self.deals = deals_df.copy()
        self.activities = activities_df.copy()
        self.contacts = contacts_df.copy()
        self.transitions = stage_transitions_df.copy()

        # Parse dates
        self.deals["created_date"] = pd.to_datetime(self.deals["created_date"])
        if "close_date" in self.deals.columns:
            self.deals["close_date"] = pd.to_datetime(
                self.deals["close_date"], errors="coerce"
            )
        self.activities["timestamp"] = pd.to_datetime(self.activities["timestamp"])
        self.contacts["first_contact_date"] = pd.to_datetime(
            self.contacts["first_contact_date"]
        )
        self.transitions["transition_date"] = pd.to_datetime(
            self.transitions["transition_date"]
        )

        # Reference date for computing "current" features (e.g., silence gap)
        if reference_date:
            self.reference_date = pd.to_datetime(reference_date)
        else:
            self.reference_date = self.activities["timestamp"].max()

    def compute_all_features(self) -> pd.DataFrame:
        """Compute all temporal features — one row per deal."""
        features = []

        for _, deal in self.deals.iterrows():
            deal_id = deal["deal_id"]
            deal_activities = self.activities[
                self.activities["deal_id"] == deal_id
            ]
            deal_contacts = self.contacts[
                self.contacts["deal_id"] == deal_id
            ]
            deal_transitions = self.transitions[
                self.transitions["deal_id"] == deal_id
            ]

            row = {"deal_id": deal_id}

            # Core features
            row.update(self._compute_response_time(deal, deal_activities))
            row.update(
                self._compute_engagement_score(deal, deal_activities)
            )
            row.update(
                self._compute_deal_velocity(deal, deal_transitions)
            )
            row.update(
                self._compute_silence_gap(deal, deal_activities)
            )
            row.update(
                self._compute_stakeholder_coverage(deal, deal_contacts)
            )
            row.update(
                self._compute_decay_weighted_engagement(deal, deal_activities)
            )

            # Add deal metadata for downstream use
            row["deal_value"] = deal["deal_value"]
            row["deal_size_bucket"] = deal["deal_size_bucket"]
            row["industry"] = deal["industry"]
            row["source"] = deal["source"]
            row["outcome"] = deal["outcome"]
            row["stage"] = deal["stage"]
            row["duration_days"] = deal["duration_days"]

            features.append(row)

        features_df = pd.DataFrame(features)

        # Compute cohort-relative velocity after all individual velocities exist
        features_df = self._add_cohort_velocity(features_df)

        return features_df

    def _compute_response_time(
        self, deal: pd.Series, activities: pd.DataFrame
    ) -> dict:
        """Average delay between outbound contact and inbound reply."""
        outbound = activities[
            activities["direction"] == "outbound"
        ].sort_values("timestamp")
        inbound = activities[
            activities["direction"] == "inbound"
        ].sort_values("timestamp")

        if outbound.empty or inbound.empty:
            return {
                "avg_response_time_hours": np.nan,
                "median_response_time_hours": np.nan,
                "response_count": 0,
            }

        response_times = []
        window = pd.Timedelta(days=RESPONSE_TIME_WINDOW_DAYS)

        for _, out_act in outbound.iterrows():
            # Find the first inbound activity after this outbound, within the window
            candidates = inbound[
                (inbound["timestamp"] > out_act["timestamp"])
                & (inbound["timestamp"] <= out_act["timestamp"] + window)
            ]
            if not candidates.empty:
                first_reply = candidates.iloc[0]
                delta = (
                    first_reply["timestamp"] - out_act["timestamp"]
                ).total_seconds() / 3600
                response_times.append(delta)

        if not response_times:
            return {
                "avg_response_time_hours": np.nan,
                "median_response_time_hours": np.nan,
                "response_count": 0,
            }

        return {
            "avg_response_time_hours": round(np.mean(response_times), 2),
            "median_response_time_hours": round(np.median(response_times), 2),
            "response_count": len(response_times),
        }

    def _compute_engagement_score(
        self, deal: pd.Series, activities: pd.DataFrame
    ) -> dict:
        """Weighted engagement score normalized by deal age in weeks."""
        if activities.empty:
            return {
                "engagement_score": 0.0,
                "engagement_per_week": 0.0,
                "total_activities": 0,
            }

        # Count activities by type
        weighted_sum = 0.0
        for activity_type, weight in ENGAGEMENT_WEIGHTS.items():
            count = len(
                activities[activities["activity_type"] == activity_type]
            )
            weighted_sum += count * weight

        # Normalize by deal age in weeks
        deal_age_days = deal["duration_days"]
        deal_age_weeks = max(1, deal_age_days / 7)

        return {
            "engagement_score": round(weighted_sum, 3),
            "engagement_per_week": round(weighted_sum / deal_age_weeks, 3),
            "total_activities": len(activities),
        }

    def _compute_deal_velocity(
        self, deal: pd.Series, transitions: pd.DataFrame
    ) -> dict:
        """Average days per stage transition."""
        if transitions.empty:
            return {
                "avg_days_per_stage": np.nan,
                "total_transitions": 0,
            }

        avg_days = transitions["days_in_previous_stage"].mean()

        return {
            "avg_days_per_stage": round(avg_days, 2),
            "total_transitions": len(transitions),
        }

    def _add_cohort_velocity(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add cohort-relative deal velocity after all deals are computed."""
        features_df["deal_velocity_ratio"] = np.nan

        for idx, row in features_df.iterrows():
            if pd.isna(row["avg_days_per_stage"]):
                continue

            # Find cohort: same size bucket + industry
            cohort = features_df[
                (features_df["deal_size_bucket"] == row["deal_size_bucket"])
                & (features_df["industry"] == row["industry"])
                & features_df["avg_days_per_stage"].notna()
            ]

            # Fall back to broader cohort if too small
            if len(cohort) < MIN_COHORT_SIZE:
                cohort = features_df[
                    (features_df["deal_size_bucket"] == row["deal_size_bucket"])
                    & features_df["avg_days_per_stage"].notna()
                ]

            if len(cohort) < 2:
                cohort = features_df[
                    features_df["avg_days_per_stage"].notna()
                ]

            cohort_median = cohort["avg_days_per_stage"].median()
            if cohort_median > 0 and row["avg_days_per_stage"] > 0:
                # Ratio > 1.0 means faster than cohort
                features_df.at[idx, "deal_velocity_ratio"] = round(
                    cohort_median / row["avg_days_per_stage"], 3
                )

        return features_df

    def _compute_silence_gap(
        self, deal: pd.Series, activities: pd.DataFrame
    ) -> dict:
        """
        Days since last meaningful inbound activity, plus contextual severity.

        Key insight: a 7-day gap in week 1 is noise. A 7-day gap in week 8
        of a previously active deal is a strong churn signal. Severity captures
        this by comparing current gap to the deal's historical baseline.
        """
        meaningful = activities[
            activities["activity_type"].isin(MEANINGFUL_ACTIVITY_TYPES)
        ].sort_values("timestamp")

        if meaningful.empty:
            return {
                "silence_gap_days": np.nan,
                "silence_gap_severity": np.nan,
                "last_meaningful_activity": None,
            }

        # Days since last meaningful activity
        last_activity = meaningful["timestamp"].max()
        end_date = (
            pd.to_datetime(deal["close_date"])
            if pd.notna(deal.get("close_date"))
            else self.reference_date
        )
        silence_days = (end_date - last_activity).days

        # Compute contextual severity
        severity = self._contextual_silence_severity(
            silence_days, deal, meaningful
        )

        return {
            "silence_gap_days": max(0, silence_days),
            "silence_gap_severity": round(severity, 3),
            "last_meaningful_activity": last_activity.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }

    def _contextual_silence_severity(
        self,
        gap_days: float,
        deal: pd.Series,
        meaningful_activities: pd.DataFrame,
    ) -> float:
        """
        Severity = (gap / deal_age_factor) * (baseline_freq / current_freq).

        Baseline frequency: avg inter-activity days during first N weeks.
        Current frequency: avg inter-activity days in last 14 days.
        """
        if len(meaningful_activities) < 2:
            return gap_days / 7.0  # Simple fallback for sparse deals

        timestamps = meaningful_activities["timestamp"].sort_values()
        deal_created = deal["created_date"]
        deal_age_weeks = max(1, deal["duration_days"] / 7)

        # Baseline: inter-activity intervals during first SILENCE_BASELINE_WEEKS weeks
        baseline_cutoff = deal_created + pd.Timedelta(
            weeks=SILENCE_BASELINE_WEEKS
        )
        baseline_activities = timestamps[timestamps <= baseline_cutoff]

        if len(baseline_activities) >= 2:
            baseline_gaps = baseline_activities.diff().dropna().dt.days
            baseline_freq = baseline_gaps.mean()
        else:
            # Not enough baseline data — use overall average
            all_gaps = timestamps.diff().dropna().dt.days
            baseline_freq = all_gaps.mean()

        baseline_freq = max(1.0, baseline_freq)

        # Current: inter-activity intervals in last 14 days
        recent_cutoff = timestamps.max() - pd.Timedelta(days=14)
        recent_activities = timestamps[timestamps >= recent_cutoff]

        if len(recent_activities) >= 2:
            recent_gaps = recent_activities.diff().dropna().dt.days
            current_freq = max(1.0, recent_gaps.mean())
        else:
            current_freq = max(1.0, gap_days)

        # Severity combines the raw gap with the ratio of frequencies
        frequency_ratio = current_freq / baseline_freq
        age_factor = max(1.0, np.log1p(deal_age_weeks))
        severity = (gap_days / age_factor) * frequency_ratio

        return severity

    def _compute_stakeholder_coverage(
        self, deal: pd.Series, contacts: pd.DataFrame
    ) -> dict:
        """Count stakeholders, check for economic buyer, compute breadth."""
        if contacts.empty:
            return {
                "stakeholder_count": 0,
                "has_economic_buyer": False,
                "decision_maker_ratio": 0.0,
                "stakeholder_breadth": 0,
            }

        n_contacts = len(contacts)
        has_eb = (contacts["role"] == "economic_buyer").any()
        n_decision_makers = contacts["is_decision_maker"].sum()
        n_seniority_levels = contacts["seniority_level"].nunique()

        return {
            "stakeholder_count": n_contacts,
            "has_economic_buyer": bool(has_eb),
            "decision_maker_ratio": round(n_decision_makers / n_contacts, 3),
            "stakeholder_breadth": n_seniority_levels,
        }

    def _compute_decay_weighted_engagement(
        self, deal: pd.Series, activities: pd.DataFrame
    ) -> dict:
        """
        Each activity weighted by exp(-lambda * days_ago).
        Half-life ~14 days — activities from 4+ weeks ago contribute <25%.
        """
        if activities.empty:
            return {"decay_weighted_engagement": 0.0}

        end_date = (
            pd.to_datetime(deal["close_date"])
            if pd.notna(deal.get("close_date"))
            else self.reference_date
        )

        days_ago = (end_date - activities["timestamp"]).dt.days
        weights = np.exp(-DECAY_LAMBDA * days_ago.values)

        # Apply activity-type base weights
        base_weights = activities["activity_type"].map(
            lambda t: ENGAGEMENT_WEIGHTS.get(t, 0.15)
        )

        total = (weights * base_weights.values).sum()

        return {"decay_weighted_engagement": round(total, 3)}
