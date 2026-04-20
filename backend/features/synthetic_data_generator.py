"""
Synthetic CRM data generator.

Produces realistic deal, contact, activity, and stage transition data
that mimics real sales pipeline behavior. Outcome-correlated patterns
are embedded so the ML model has learnable signal, with enough noise
to make the prediction problem non-trivial (~70-80% AUC target).
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict

from backend.features.feature_config import (
    STAGE_ORDER,
    STAGE_DURATION_MEDIAN,
    DEAL_SIZE_RANGES,
    INDUSTRIES,
    LEAD_SOURCES,
    CONTACT_ROLES,
    SENIORITY_LEVELS,
    REP_NAMES,
)


class SyntheticCRMGenerator:
    """Generates synthetic CRM data with realistic temporal patterns."""

    def __init__(self, n_deals: int = 75, seed: int = 42):
        self.n_deals = n_deals
        self.rng = np.random.RandomState(seed)
        self.base_date = datetime(2024, 1, 15)

        self.deals_df = None
        self.contacts_df = None
        self.activities_df = None
        self.stage_transitions_df = None

    def generate_all(self) -> Dict[str, pd.DataFrame]:
        """Run the full generation pipeline and return all DataFrames."""
        self.deals_df = self._generate_deals()
        self.contacts_df = self._generate_contacts()
        self.stage_transitions_df = self._generate_stage_transitions()
        self.activities_df = self._generate_activities()

        return {
            "deals": self.deals_df,
            "contacts": self.contacts_df,
            "activities": self.activities_df,
            "stage_transitions": self.stage_transitions_df,
        }

    def _generate_deals(self) -> pd.DataFrame:
        """Generate core deal records with outcome-correlated properties."""
        deals = []

        for i in range(self.n_deals):
            deal_id = f"DEAL-{i+1:04d}"

            # Determine outcome first — it drives everything else
            outcome_roll = self.rng.random()
            if outcome_roll < 0.30:
                outcome = "won"
            elif outcome_roll < 0.70:
                outcome = "lost"
            else:
                outcome = "active"

            # Deal properties
            size_bucket = self.rng.choice(
                list(DEAL_SIZE_RANGES.keys()), p=[0.50, 0.35, 0.15]
            )
            value_range = DEAL_SIZE_RANGES[size_bucket]
            deal_value = round(
                self.rng.uniform(value_range[0], value_range[1]), -2
            )

            industry = self.rng.choice(INDUSTRIES)
            source = self.rng.choice(LEAD_SOURCES, p=[0.40, 0.35, 0.25])
            owner_rep = self.rng.choice(REP_NAMES)

            # Created date: spread across 6 months
            days_offset = self.rng.randint(0, 180)
            created_date = self.base_date + timedelta(days=int(days_offset))

            # Deal duration depends on outcome and size
            base_duration = {"smb": 30, "mid_market": 50, "enterprise": 80}
            if outcome == "won":
                duration = max(
                    15,
                    int(
                        self.rng.lognormal(
                            np.log(base_duration[size_bucket] * 0.9), 0.3
                        )
                    ),
                )
            elif outcome == "lost":
                duration = max(
                    10,
                    int(
                        self.rng.lognormal(
                            np.log(base_duration[size_bucket] * 0.7), 0.4
                        )
                    ),
                )
            else:  # active
                duration = max(
                    7,
                    int(
                        self.rng.lognormal(
                            np.log(base_duration[size_bucket] * 0.5), 0.3
                        )
                    ),
                )

            close_date = (
                created_date + timedelta(days=duration)
                if outcome != "active"
                else None
            )

            # Determine current stage
            if outcome == "won":
                stage = "closed_won"
            elif outcome == "lost":
                stage = "closed_lost"
            else:
                active_stages = STAGE_ORDER[:4]
                stage = self.rng.choice(active_stages)

            deals.append(
                {
                    "deal_id": deal_id,
                    "company_name": f"Company_{self.rng.randint(100, 999)}",
                    "deal_value": deal_value,
                    "stage": stage,
                    "created_date": created_date.strftime("%Y-%m-%d"),
                    "close_date": (
                        close_date.strftime("%Y-%m-%d") if close_date else None
                    ),
                    "outcome": outcome,
                    "owner_rep": owner_rep,
                    "industry": industry,
                    "deal_size_bucket": size_bucket,
                    "source": source,
                    "duration_days": duration,
                }
            )

        return pd.DataFrame(deals)

    def _generate_contacts(self) -> pd.DataFrame:
        """Generate contacts per deal — won deals get more stakeholders."""
        contacts = []
        contact_counter = 0

        for _, deal in self.deals_df.iterrows():
            # Won deals: more contacts, especially decision-makers
            if deal["outcome"] == "won":
                n_contacts = self.rng.randint(3, 7)
                has_econ_buyer = self.rng.random() < 0.85
            elif deal["outcome"] == "lost":
                n_contacts = self.rng.randint(1, 4)
                has_econ_buyer = self.rng.random() < 0.30
            else:
                n_contacts = self.rng.randint(1, 5)
                has_econ_buyer = self.rng.random() < 0.50

            roles_assigned = []
            for j in range(n_contacts):
                contact_counter += 1

                if j == 0:
                    role = "champion"
                elif j == 1 and has_econ_buyer:
                    role = "economic_buyer"
                else:
                    role = self.rng.choice(
                        [r for r in CONTACT_ROLES if r != "champion"]
                    )
                roles_assigned.append(role)

                seniority = self._role_to_seniority(role)
                is_decision_maker = role in ["economic_buyer", "champion"]

                # First contact date: near deal creation
                days_after_creation = self.rng.randint(0, min(7, deal["duration_days"]))
                first_contact = pd.to_datetime(
                    deal["created_date"]
                ) + timedelta(days=int(days_after_creation))

                contacts.append(
                    {
                        "contact_id": f"CONTACT-{contact_counter:05d}",
                        "deal_id": deal["deal_id"],
                        "name": f"Contact_{contact_counter}",
                        "role": role,
                        "seniority_level": seniority,
                        "first_contact_date": first_contact.strftime("%Y-%m-%d"),
                        "is_decision_maker": is_decision_maker,
                    }
                )

        return pd.DataFrame(contacts)

    def _role_to_seniority(self, role: str) -> str:
        """Map contact role to likely seniority level."""
        seniority_map = {
            "economic_buyer": ["c_level", "vp"],
            "champion": ["director", "manager"],
            "technical_evaluator": ["manager", "individual_contributor"],
            "end_user": ["individual_contributor", "manager"],
            "blocker": ["vp", "director", "manager"],
        }
        return self.rng.choice(seniority_map.get(role, SENIORITY_LEVELS))

    def _generate_stage_transitions(self) -> pd.DataFrame:
        """Generate stage progression history for each deal."""
        transitions = []
        transition_counter = 0

        for _, deal in self.deals_df.iterrows():
            created = pd.to_datetime(deal["created_date"])
            duration = deal["duration_days"]
            outcome = deal["outcome"]

            # Determine which stages this deal passed through
            if outcome == "won":
                final_stage_idx = STAGE_ORDER.index("closed_won")
            elif outcome == "lost":
                # Lost deals die at various stages
                final_stage_idx = self.rng.randint(1, 4)
                # Replace final stage with closed_lost
            else:
                final_stage_idx = STAGE_ORDER.index(deal["stage"])

            stages_traversed = STAGE_ORDER[:final_stage_idx]
            if outcome == "lost":
                stages_traversed.append("closed_lost")
            elif outcome == "won":
                stages_traversed.append("closed_won")

            if len(stages_traversed) < 2:
                continue

            # Distribute duration across stages
            n_transitions = len(stages_traversed) - 1
            raw_durations = []
            for s in stages_traversed[:-1]:
                median = STAGE_DURATION_MEDIAN.get(s, 10)
                if outcome == "won":
                    d = max(2, int(self.rng.lognormal(np.log(median * 0.8), 0.3)))
                else:
                    d = max(2, int(self.rng.lognormal(np.log(median * 1.2), 0.4)))
                raw_durations.append(d)

            # Scale durations to fit within deal duration
            total_raw = sum(raw_durations)
            if total_raw > 0:
                scale_factor = duration / total_raw
                scaled_durations = [
                    max(1, int(d * scale_factor)) for d in raw_durations
                ]
            else:
                scaled_durations = raw_durations

            current_date = created
            for k in range(n_transitions):
                transition_counter += 1
                days_in_stage = scaled_durations[k] if k < len(scaled_durations) else 7
                next_date = current_date + timedelta(days=days_in_stage)

                transitions.append(
                    {
                        "transition_id": f"TRANS-{transition_counter:05d}",
                        "deal_id": deal["deal_id"],
                        "from_stage": stages_traversed[k],
                        "to_stage": stages_traversed[k + 1],
                        "transition_date": next_date.strftime("%Y-%m-%d"),
                        "days_in_previous_stage": days_in_stage,
                    }
                )
                current_date = next_date

        return pd.DataFrame(transitions)

    def _generate_activities(self) -> pd.DataFrame:
        """Generate temporal activity log with outcome-correlated patterns."""
        activities = []
        activity_counter = 0

        for _, deal in self.deals_df.iterrows():
            deal_contacts = self.contacts_df[
                self.contacts_df["deal_id"] == deal["deal_id"]
            ]

            if deal_contacts.empty:
                continue

            created = pd.to_datetime(deal["created_date"])
            duration = deal["duration_days"]
            outcome = deal["outcome"]

            # Activity frequency depends on outcome
            if outcome == "won":
                base_freq = self.rng.uniform(0.8, 1.5)  # activities per day
                inbound_ratio = 0.45  # higher inbound for won deals
            elif outcome == "lost":
                base_freq = self.rng.uniform(0.3, 0.8)
                inbound_ratio = 0.20  # lower inbound for lost deals
            else:
                base_freq = self.rng.uniform(0.4, 1.0)
                inbound_ratio = 0.30

            # Simulate rep logging inconsistency
            logging_consistency = self.rng.uniform(0.5, 1.0)
            effective_freq = base_freq * logging_consistency

            # Generate activities day by day
            for day in range(duration):
                current_date = created + timedelta(days=day)

                # Skip weekends
                if current_date.weekday() >= 5:
                    continue

                # Activity frequency decay for lost deals
                if outcome == "lost":
                    decay = np.exp(-0.03 * day)
                    day_freq = effective_freq * decay
                elif outcome == "won":
                    # Won deals may have activity bursts near close
                    days_to_close = duration - day
                    if days_to_close < 10:
                        day_freq = effective_freq * 1.5
                    else:
                        day_freq = effective_freq
                else:
                    day_freq = effective_freq

                # Number of activities this day (Poisson-distributed)
                n_activities = self.rng.poisson(day_freq)

                for _ in range(n_activities):
                    activity_counter += 1
                    contact = deal_contacts.sample(1, random_state=self.rng).iloc[0]

                    # Determine direction and type
                    is_inbound = self.rng.random() < inbound_ratio
                    activity_type = self._pick_activity_type(
                        day, duration, is_inbound
                    )
                    direction = "inbound" if is_inbound else "outbound"

                    # Timestamp within business hours
                    hour = self.rng.randint(8, 18)
                    minute = self.rng.randint(0, 59)
                    timestamp = current_date.replace(hour=hour, minute=minute)

                    # Sentiment: inbound on won deals trends positive
                    if is_inbound and outcome == "won":
                        sentiment = min(1.0, self.rng.beta(6, 3))
                    elif is_inbound and outcome == "lost":
                        sentiment = min(1.0, self.rng.beta(3, 5))
                    else:
                        sentiment = min(1.0, self.rng.beta(4, 4))

                    activities.append(
                        {
                            "activity_id": f"ACT-{activity_counter:06d}",
                            "deal_id": deal["deal_id"],
                            "contact_id": contact["contact_id"],
                            "activity_type": activity_type,
                            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                            "direction": direction,
                            "sentiment_score": round(sentiment, 3),
                        }
                    )

        return pd.DataFrame(activities)

    def _pick_activity_type(
        self, day: int, total_days: int, is_inbound: bool
    ) -> str:
        """Pick activity type appropriate to deal stage and direction."""
        progress = day / max(1, total_days)

        if is_inbound:
            if progress < 0.3:
                types = ["email_received", "call_inbound"]
                probs = [0.7, 0.3]
            elif progress < 0.6:
                types = ["email_received", "call_inbound", "meeting"]
                probs = [0.4, 0.2, 0.4]
            else:
                types = ["email_received", "call_inbound", "meeting"]
                probs = [0.3, 0.2, 0.5]
        else:
            if progress < 0.3:
                types = ["email_sent", "call_outbound"]
                probs = [0.6, 0.4]
            elif progress < 0.6:
                types = ["email_sent", "call_outbound", "demo", "proposal_sent"]
                probs = [0.3, 0.2, 0.3, 0.2]
            else:
                types = ["email_sent", "call_outbound", "proposal_sent"]
                probs = [0.4, 0.3, 0.3]

        return self.rng.choice(types, p=probs)

    def save_to_csv(self, output_dir: str = "data/raw") -> None:
        """Save all generated DataFrames to CSV files."""
        import os

        os.makedirs(output_dir, exist_ok=True)

        if self.deals_df is None:
            raise RuntimeError("Call generate_all() before saving.")

        self.deals_df.to_csv(f"{output_dir}/deals.csv", index=False)
        self.contacts_df.to_csv(f"{output_dir}/contacts.csv", index=False)
        self.activities_df.to_csv(f"{output_dir}/activities.csv", index=False)
        self.stage_transitions_df.to_csv(
            f"{output_dir}/stage_transitions.csv", index=False
        )
        print(f"Saved {len(self.deals_df)} deals to {output_dir}/")
        print(f"  - {len(self.contacts_df)} contacts")
        print(f"  - {len(self.activities_df)} activities")
        print(f"  - {len(self.stage_transitions_df)} stage transitions")
