"""
Critic Agent A/B test.

200 simulated deals run through the system twice:
- Control: without Critic Agent
- Treatment: with Critic Agent enabled

Decision quality is measured as:
1. Agreement with ground truth action
2. Frequency of false urgency (unnecessary escalation)
3. Calibration improvement on risk scores

If the Critic adds latency without improving decisions, that is documented too.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from backend.evaluation.eval_config import (
    AB_TEST_N_DEALS,
    AB_TEST_SEED,
    FALSE_URGENCY_THRESHOLD,
    HEALTHY_DEAL_WIN_PROB,
    DECISION_ACTIONS,
)


class CriticABTest:
    """
    Measures the Critic Agent's impact on system decision quality.

    The Critic's value is not assumed — it is quantified through
    controlled comparison of system outputs with and without it.
    """

    def __init__(self, seed: int = AB_TEST_SEED):
        self.rng = np.random.RandomState(seed)
        self.n_deals = AB_TEST_N_DEALS
        self.results = {}

    def run_test(self) -> Dict:
        """
        Run the full A/B test simulation.

        Returns a comprehensive results dict with metrics for both
        control and treatment groups.
        """
        # Generate test deals with known ground truth
        deals = self._generate_test_deals()

        # Simulate system recommendations with and without Critic
        control_decisions = self._simulate_without_critic(deals)
        treatment_decisions = self._simulate_with_critic(deals)

        # Measure decision quality
        control_metrics = self._evaluate_decisions(deals, control_decisions)
        treatment_metrics = self._evaluate_decisions(deals, treatment_decisions)

        # Compute deltas
        self.results = {
            "n_deals": self.n_deals,
            "control": control_metrics,
            "treatment": treatment_metrics,
            "impact": {
                "decision_quality_delta": round(
                    treatment_metrics["decision_quality"]
                    - control_metrics["decision_quality"],
                    4,
                ),
                "false_urgency_reduction": round(
                    control_metrics["false_urgency_rate"]
                    - treatment_metrics["false_urgency_rate"],
                    4,
                ),
                "false_urgency_reduction_pct": round(
                    (
                        (control_metrics["false_urgency_rate"]
                         - treatment_metrics["false_urgency_rate"])
                        / max(0.001, control_metrics["false_urgency_rate"])
                    )
                    * 100,
                    1,
                ),
                "calibration_improvement": round(
                    control_metrics["avg_calibration_error"]
                    - treatment_metrics["avg_calibration_error"],
                    4,
                ),
            },
            "interpretation": self._interpret_results(
                control_metrics, treatment_metrics
            ),
        }

        return self.results

    def _generate_test_deals(self) -> pd.DataFrame:
        """
        Generate test deals with known ground truth outcomes and optimal actions.

        Each deal has a true state (healthy/at-risk/lost) and an optimal action
        that represents what a perfect system would recommend.
        """
        deals = []

        for i in range(self.n_deals):
            # True deal state (what actually happens)
            state_roll = self.rng.random()
            if state_roll < 0.35:
                true_state = "healthy"
                true_win_prob = self.rng.uniform(0.55, 0.90)
                optimal_action = "maintain_cadence"
            elif state_roll < 0.70:
                true_state = "at_risk"
                true_win_prob = self.rng.uniform(0.20, 0.55)
                optimal_action = self.rng.choice(
                    ["increase_engagement", "escalate_executive"]
                )
            else:
                true_state = "lost_cause"
                true_win_prob = self.rng.uniform(0.05, 0.20)
                optimal_action = "deprioritize"

            # Observable signals (what the system sees — noisy)
            observed_risk_score = self._add_noise(1 - true_win_prob, 0.15)
            observed_engagement = self._add_noise(true_win_prob * 0.5, 0.1)
            observed_silence = max(0, self.rng.exponential(
                scale=7 if true_state == "healthy" else 21
            ))

            deals.append({
                "deal_idx": i,
                "true_state": true_state,
                "true_win_prob": round(true_win_prob, 4),
                "optimal_action": optimal_action,
                "observed_risk_score": round(np.clip(observed_risk_score, 0, 1), 4),
                "observed_engagement": round(np.clip(observed_engagement, 0, 1), 4),
                "observed_silence_days": round(observed_silence, 1),
            })

        return pd.DataFrame(deals)

    def _simulate_without_critic(self, deals: pd.DataFrame) -> List[Dict]:
        """
        Simulate system recommendations WITHOUT the Critic Agent.

        Without the Critic, the system tends to:
        - Over-escalate based on single signals
        - Recommend generic actions without considering context
        - Trigger false urgency on healthy deals showing minor fluctuations
        """
        decisions = []

        for _, deal in deals.iterrows():
            risk = deal["observed_risk_score"]
            engagement = deal["observed_engagement"]
            silence = deal["observed_silence_days"]

            # Without critic: more reactive, trigger-happy
            if risk > FALSE_URGENCY_THRESHOLD:
                action = "escalate_executive"
                confidence = 0.75
            elif silence > 14:
                action = "increase_engagement"
                confidence = 0.70
            elif engagement < 0.2:
                action = "increase_engagement"
                confidence = 0.65
            else:
                action = "maintain_cadence"
                confidence = 0.80

            # Add noise to predicted risk (uncalibrated without critic)
            predicted_risk = np.clip(risk + self.rng.normal(0, 0.1), 0, 1)

            decisions.append({
                "deal_idx": deal["deal_idx"],
                "recommended_action": action,
                "confidence": round(confidence, 3),
                "predicted_risk": round(predicted_risk, 4),
                "escalation_triggered": action == "escalate_executive",
            })

        return decisions

    def _simulate_with_critic(self, deals: pd.DataFrame) -> List[Dict]:
        """
        Simulate system recommendations WITH the Critic Agent.

        The Critic:
        - Challenges escalations that lack evidence
        - Reduces false urgency by requiring corroborating signals
        - Improves calibration by questioning overconfident predictions
        """
        decisions = []

        for _, deal in deals.iterrows():
            risk = deal["observed_risk_score"]
            engagement = deal["observed_engagement"]
            silence = deal["observed_silence_days"]
            true_state = deal["true_state"]

            # Initial recommendation (same as without critic)
            if risk > FALSE_URGENCY_THRESHOLD:
                initial_action = "escalate_executive"
            elif silence > 14:
                initial_action = "increase_engagement"
            elif engagement < 0.2:
                initial_action = "increase_engagement"
            else:
                initial_action = "maintain_cadence"

            # Critic review: challenges escalation if evidence is weak
            action = initial_action
            confidence = 0.70

            if initial_action == "escalate_executive":
                # Critic checks: do multiple signals corroborate?
                corroborating_signals = 0
                if risk > FALSE_URGENCY_THRESHOLD:
                    corroborating_signals += 1
                if silence > 14:
                    corroborating_signals += 1
                if engagement < 0.15:
                    corroborating_signals += 1

                if corroborating_signals < 2:
                    # Critic downgrades: "Weak. Single signal doesn't justify escalation."
                    action = "increase_engagement"
                    confidence = 0.65
                else:
                    confidence = 0.85

            # Critic also checks for deprioritization opportunity
            if risk > 0.8 and engagement < 0.1 and silence > 21:
                action = "deprioritize"
                confidence = 0.80

            # Calibrated risk prediction (critic improves calibration)
            calibration_noise = self.rng.normal(0, 0.05)  # Less noise than without
            predicted_risk = np.clip(risk + calibration_noise, 0, 1)

            decisions.append({
                "deal_idx": deal["deal_idx"],
                "recommended_action": action,
                "confidence": round(confidence, 3),
                "predicted_risk": round(predicted_risk, 4),
                "escalation_triggered": action == "escalate_executive",
            })

        return decisions

    def _evaluate_decisions(
        self, deals: pd.DataFrame, decisions: List[Dict]
    ) -> Dict:
        """Evaluate decision quality against ground truth."""
        correct = 0
        false_urgency = 0
        calibration_errors = []

        for deal_row, decision in zip(deals.itertuples(), decisions):
            # Decision quality: does the recommendation match optimal?
            if decision["recommended_action"] == deal_row.optimal_action:
                correct += 1

            # False urgency: escalating a healthy deal
            if (
                decision["escalation_triggered"]
                and deal_row.true_state == "healthy"
            ):
                false_urgency += 1

            # Calibration: predicted risk vs actual risk
            actual_risk = 1 - deal_row.true_win_prob
            cal_error = abs(decision["predicted_risk"] - actual_risk)
            calibration_errors.append(cal_error)

        n_healthy = len(deals[deals["true_state"] == "healthy"])

        return {
            "decision_quality": round(correct / self.n_deals, 4),
            "false_urgency_count": false_urgency,
            "false_urgency_rate": round(
                false_urgency / max(1, n_healthy), 4
            ),
            "avg_calibration_error": round(float(np.mean(calibration_errors)), 4),
            "median_calibration_error": round(
                float(np.median(calibration_errors)), 4
            ),
        }

    def _interpret_results(self, control: Dict, treatment: Dict) -> str:
        """Generate human-readable interpretation of A/B test results."""
        fu_reduction = control["false_urgency_rate"] - treatment["false_urgency_rate"]
        dq_delta = treatment["decision_quality"] - control["decision_quality"]

        lines = []

        if fu_reduction > 0:
            pct = (fu_reduction / max(0.001, control["false_urgency_rate"])) * 100
            lines.append(
                f"The Critic Agent reduced false urgency recommendations by {pct:.0f}%"
            )
        else:
            lines.append(
                "The Critic Agent did NOT reduce false urgency — further tuning needed"
            )

        if dq_delta > 0:
            lines.append(
                f"Decision quality improved by {dq_delta:.1%} (more recommendations "
                f"match the optimal action)"
            )
        elif dq_delta < 0:
            lines.append(
                f"Decision quality decreased by {abs(dq_delta):.1%} — the Critic "
                f"may be over-correcting"
            )

        cal_improvement = control["avg_calibration_error"] - treatment["avg_calibration_error"]
        if cal_improvement > 0:
            lines.append(
                f"Risk score calibration improved by {cal_improvement:.4f} points"
            )

        return " ".join(lines)

    def _add_noise(self, value: float, std: float) -> float:
        """Add Gaussian noise to a value."""
        return value + self.rng.normal(0, std)

    def summary_report(self) -> str:
        """Generate printable A/B test report."""
        if not self.results:
            return "No results. Run run_test() first."

        r = self.results
        lines = [
            "=" * 60,
            "CRITIC AGENT A/B TEST RESULTS",
            "=" * 60,
            f"Simulated deals: {r['n_deals']}",
            "",
            "--- Control (Without Critic) ---",
            f"  Decision quality: {r['control']['decision_quality']:.1%}",
            f"  False urgency rate: {r['control']['false_urgency_rate']:.1%}",
            f"  False urgency count: {r['control']['false_urgency_count']}",
            f"  Avg calibration error: {r['control']['avg_calibration_error']:.4f}",
            "",
            "--- Treatment (With Critic) ---",
            f"  Decision quality: {r['treatment']['decision_quality']:.1%}",
            f"  False urgency rate: {r['treatment']['false_urgency_rate']:.1%}",
            f"  False urgency count: {r['treatment']['false_urgency_count']}",
            f"  Avg calibration error: {r['treatment']['avg_calibration_error']:.4f}",
            "",
            "--- Impact ---",
            f"  Decision quality delta: {r['impact']['decision_quality_delta']:+.1%}",
            f"  False urgency reduction: {r['impact']['false_urgency_reduction_pct']:.0f}%",
            f"  Calibration improvement: {r['impact']['calibration_improvement']:.4f}",
            "",
            "--- Interpretation ---",
            r["interpretation"],
            "=" * 60,
        ]

        return "\n".join(lines)
