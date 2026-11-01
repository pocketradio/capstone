"""Small deterministic checks for split, policy, and feedback invariants."""

import unittest

import numpy as np
import pandas as pd

from fraudlab.bandit import BanditConfig, LinearContextualBandit, make_context
from fraudlab.config import CostConfig, FeedbackConfig, SplitConfig
from fraudlab.data import chronological_positions
from fraudlab.features import add_causal_features
from fraudlab.models import _prepare_xgb_frames
from fraudlab.metrics import APPROVE, BLOCK, REVIEW, classification_metrics, decision_cost
from fraudlab.policy import threshold_actions, tune_threshold_policy
from fraudlab.simulator import _schedule_release, run_bandit_simulation


class CoreInvariantTests(unittest.TestCase):
    def test_chronological_split_keeps_expected_periods(self):
        data = pd.DataFrame({"TransactionDT": np.arange(100)})
        parts = chronological_positions(data, SplitConfig())
        self.assertEqual([len(parts[k]) for k in ("fit", "validation", "simulation")], [48, 12, 40])
        self.assertLess(data.iloc[parts["fit"]].TransactionDT.max(), data.iloc[parts["validation"]].TransactionDT.min())
        self.assertLess(data.iloc[parts["validation"]].TransactionDT.max(), data.iloc[parts["simulation"]].TransactionDT.min())

    def test_cost_table(self):
        costs = CostConfig()
        self.assertEqual(decision_cost(APPROVE, 1, 20.0, costs), 20.0)
        self.assertEqual(decision_cost(APPROVE, 0, 20.0, costs), 0.0)
        self.assertEqual(decision_cost(BLOCK, 1, 20.0, costs), 0.0)
        self.assertAlmostEqual(decision_cost(BLOCK, 0, 20.0, costs), 2.0)
        self.assertEqual(decision_cost(REVIEW, 1, 20.0, costs), 5.0)

    def test_ranking_metrics_and_threshold_tuning(self):
        metrics = classification_metrics([0, 1, 0, 1], [0.1, 0.9, 0.3, 0.8])
        self.assertAlmostEqual(metrics["pr_auc"], 1.0)
        self.assertAlmostEqual(metrics["roc_auc"], 1.0)
        policy = tune_threshold_policy(
            [0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1], [50.0] * 4, [1, 1, 2, 2],
            CostConfig(review_cap=0.5), grid_size=5,
        )
        self.assertLessEqual(policy["t_review"], policy["t_block"])

    def test_threshold_policy_obeys_daily_cap(self):
        costs = CostConfig(review_cap=0.03)
        p = np.linspace(0.0, 1.0, 100)
        actions = threshold_actions(p, np.full(100, 50.0), 0.2, 0.8, costs)
        self.assertLessEqual(np.count_nonzero(actions == REVIEW), 3)

    def test_simulator_cap_and_oracle_feedback(self):
        records = []
        for day in range(1, 8):
            for i in range(100):
                records.append({"isFraud": int(i % 23 == 0), "TransactionAmt": float(10 + i), "transaction_day": day})
        frame = pd.DataFrame(records)
        probabilities = np.where(frame.isFraud.to_numpy() == 1, 0.8, 0.2)
        costs = CostConfig(review_cap=0.03)
        cfg = BanditConfig(seed=7)
        delays = FeedbackConfig(
            review_delay_days=1, maturity_days=2, fraud_delay_min_days=1,
            fraud_delay_max_days=2, fraud_censor_probability=0.0, seed=7,
        )
        closed, closed_metrics = run_bandit_simulation(
            frame, probabilities, "closed_loop", costs, delays, cfg, seed=7
        )
        oracle, oracle_metrics = run_bandit_simulation(
            frame, probabilities, "oracle", costs, delays, cfg, seed=7
        )
        closed_repeat, _ = run_bandit_simulation(
            frame, probabilities, "closed_loop", costs, delays, cfg, seed=7
        )
        day_reviews = closed.groupby("day").action_id.apply(lambda x: int((x == REVIEW).sum()))
        self.assertTrue((day_reviews <= 3).all())
        self.assertEqual(oracle_metrics["feedback_updates"], len(frame))
        self.assertGreater(closed_metrics["feedback_updates"], 0)
        self.assertEqual(sum(closed_metrics["feedback_updates_by_action"].values()), closed_metrics["feedback_updates"])
        final_day = int(frame.transaction_day.max())
        released_by_horizon = int((closed.label_release_day <= final_day).fillna(False).sum())
        self.assertEqual(closed_metrics["feedback_updates"], released_by_horizon)
        self.assertTrue((closed.loc[closed.label_release_day.notna(), "label_release_day"] >
                         closed.loc[closed.label_release_day.notna(), "day"]).all())
        self.assertTrue((closed.loc[closed.action_id == BLOCK, "label_release_day"].isna()).all())
        self.assertEqual(closed_metrics["blocked_labels_scheduled"], 0)
        self.assertEqual(closed_metrics["blocked_labels_released_by_horizon"], 0)
        self.assertTrue((oracle["label_release_day"] == oracle["day"]).all())
        np.testing.assert_array_equal(closed.action_id.to_numpy(), closed_repeat.action_id.to_numpy())

    def test_blocked_feedback_hidden_or_released_after_exact_delay(self):
        rng = np.random.default_rng(1)
        hidden = FeedbackConfig(blocked_label_delay_days=None)
        delayed = FeedbackConfig(blocked_label_delay_days=30)
        self.assertIsNone(_schedule_release(BLOCK, 1, 17, hidden, rng))
        self.assertEqual(_schedule_release(BLOCK, 0, 17, delayed, rng), 47)
        self.assertEqual(_schedule_release(BLOCK, 1, 17, delayed, rng), 47)

    def test_bandit_masks_review_at_zero_capacity(self):
        policy = LinearContextualBandit(BanditConfig(seed=4))
        obs = policy.act(make_context(0.8, 100.0, 0.0, 1000.0), review_available=False)
        self.assertNotEqual(obs.action, REVIEW)
        self.assertAlmostEqual(obs.behavior_probability, 0.5)

    def test_history_features_use_only_prior_transactions(self):
        source = pd.DataFrame({
            "TransactionID": [1, 2, 3],
            "TransactionDT": [10, 20, 30],
            "TransactionAmt": [10.0, 1000.0, 30.0],
            "isFraud": [0, 1, 0],
            "card1": [7, 7, 7],
            "addr1": [5, 5, 5],
            "P_emaildomain": ["a.test", "a.test", "a.test"],
        })
        features, _, blocks, control = add_causal_features(source)
        self.assertEqual(features.loc[1, "card1_transactions_before"], 1)
        self.assertAlmostEqual(features.loc[1, "card1_amount_mean_before"], 10.0)
        self.assertAlmostEqual(features.loc[2, "card1_amount_mean_before"], 505.0)
        relabeled = source.copy()
        relabeled["isFraud"] = 1 - relabeled["isFraud"]
        features_changed, _, _, _ = add_causal_features(relabeled)
        compared = [c for c in features.columns if c != "isFraud"]
        pd.testing.assert_frame_equal(features[compared], features_changed[compared])

    def test_tied_timestamps_do_not_see_each_other(self):
        source = pd.DataFrame({
            "TransactionID": [1, 2, 3], "TransactionDT": [10, 10, 20],
            "TransactionAmt": [10.0, 30.0, 50.0], "isFraud": [0, 1, 0],
            "card1": [7, 7, 7], "addr1": [5, 5, 5], "P_emaildomain": ["a", "a", "a"],
        })
        features, _, _, _ = add_causal_features(source)
        self.assertEqual(features.loc[0, "card1_transactions_before"], 0)
        self.assertEqual(features.loc[1, "card1_transactions_before"], 0)
        self.assertTrue(pd.isna(features.loc[0, "card1_amount_mean_before"]))
        self.assertTrue(pd.isna(features.loc[1, "card1_amount_mean_before"]))
        self.assertEqual(features.loc[2, "card1_transactions_before"], 2)
        self.assertAlmostEqual(features.loc[2, "card1_amount_mean_before"], 20.0)

    def test_feature_blocks_and_d1_uid_are_created(self):
        source = pd.DataFrame({
            "TransactionID": [1, 2, 3], "TransactionDT": [86400, 172800, 259200],
            "TransactionAmt": [10.25, 20.50, 30.75], "isFraud": [0, 0, 1],
            "card1": [1, 1, 1], "card2": [2, 2, 2], "card3": [3, 3, 3],
            "addr1": [4, 4, 4], "P_emaildomain": ["a", "a", "a"], "D1": [1, 2, 3], "D9": [0.1, 0.2, 0.3],
        })
        features, _, blocks, control = add_causal_features(source)
        self.assertIn("card2_transactions_before", blocks["frequency"])
        self.assertIn("uid_transactions_before", blocks["uid"])
        self.assertTrue(any("D9_std_before" in c for c in blocks["aggregates"]))
        self.assertIn("amount_dollars", blocks["amount_parts"])
        self.assertIn("transaction_day", control)
        self.assertEqual(features.loc[1, "uid_transactions_before"], 1)
        self.assertEqual(features.loc[1, "amount_cents"], 50)

    def test_xgboost_does_not_use_static_high_cardinality_frequency_maps(self):
        fit = pd.DataFrame({"high_card": [str(i) for i in range(25)], "small_cat": ["a", "b"] * 12 + ["a"]})
        transformed, state = _prepare_xgb_frames(fit, [fit.iloc[:2]], ["high_card", "small_cat"])
        self.assertNotIn("frequency_maps", state)
        self.assertTrue(all(not c.startswith("freq_") for c in transformed[0].columns))
        self.assertFalse(any(c.startswith("high_card") for c in transformed[0].columns))


if __name__ == "__main__":
    unittest.main()

