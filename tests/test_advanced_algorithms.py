import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analytics.advanced_algorithms import (  # noqa: E402
    binary_sum_product,
    build_report_feature_matrix,
    build_supervised_report_dataset,
    evaluate_binary_classifier,
    hmm_forward_backward,
    kmeans,
    pca,
)


class AdvancedAlgorithmsTests(unittest.TestCase):
    def test_kmeans_finds_two_separated_groups_deterministically(self):
        values = np.array([[0.0], [0.1], [9.9], [10.0]])
        labels, centers, inertia = kmeans(values, 2, random_state=7)

        self.assertEqual(len(set(labels)), 2)
        self.assertEqual(centers.shape, (2, 1))
        self.assertAlmostEqual(inertia, 0.01, places=6)
        np.testing.assert_array_equal(labels, kmeans(values, 2, random_state=7)[0])

    def test_pca_returns_requested_projection_and_variance(self):
        values = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
        scores, explained = pca(values, n_components=2)

        self.assertEqual(scores.shape, (3, 2))
        self.assertAlmostEqual(float(explained.sum()), 1.0)

    def test_hmm_posteriors_are_normalized(self):
        posteriors, log_likelihood = hmm_forward_backward(
            [0, 1],
            np.array([0.8, 0.2]),
            np.array([[0.9, 0.1], [0.2, 0.8]]),
            np.array([[0.9, 0.1], [0.1, 0.9]]),
        )

        self.assertEqual(posteriors.shape, (2, 2))
        np.testing.assert_allclose(posteriors.sum(axis=1), 1.0)
        self.assertLess(log_likelihood, 0)
        self.assertGreater(posteriors[1, 1], posteriors[1, 0])

    def test_sum_product_combines_messages_and_neutral_evidence(self):
        posterior = binary_sum_product(
            0.2,
            [np.array([0.1, 0.9]), np.array([1.0, 1.0])],
        )

        np.testing.assert_allclose(posterior, [0.08 / 0.26, 0.18 / 0.26])

    def test_report_features_aggregate_multiple_images_per_report(self):
        labels = pd.DataFrame(
            {
                "report_id": ["r1", "r1", "r2"],
                "Finding": ["PRESENT", "PRESENT", "ABSENT"],
            }
        )

        ids, features, values = build_report_feature_matrix(labels, ["Finding"])

        self.assertEqual(ids, ["r1", "r2"])
        self.assertEqual(values.shape, (2, 3))
        self.assertEqual(features.iloc[0]["Finding — present"], 1.0)
        self.assertEqual(features.iloc[1]["Finding — absent"], 1.0)

    def test_hmm_accepts_numpy_observation_arrays(self):
        posteriors, log_likelihood = hmm_forward_backward(
            np.array([0, 1]),
            np.array([0.8, 0.2]),
            np.array([[0.9, 0.1], [0.2, 0.8]]),
            np.array([[0.9, 0.1], [0.1, 0.9]]),
        )

        self.assertEqual(posteriors.shape, (2, 2))
        np.testing.assert_allclose(posteriors.sum(axis=1), 1.0)
        self.assertLess(log_likelihood, 0)

    def test_build_report_feature_matrix_requires_conditions(self):
        labels = pd.DataFrame({"report_id": ["r1"], "Finding": ["PRESENT"]})

        with self.assertRaises(ValueError):
            build_report_feature_matrix(labels, [])

    def test_supervised_report_dataset_excludes_unknown_targets_and_target_feature(self):
        labels = pd.DataFrame(
            {
                "report_id": ["r1", "r2", "r3", "r4", "r5", "r6", "r7"],
                "Target": [
                    "PRESENT",
                    "ABSENT",
                    "NOT_MENTIONED",
                    "PRESENT",
                    "ABSENT",
                    "ABSENT",
                    "PRESENT",
                ],
                "Signal": [
                    "PRESENT",
                    "ABSENT",
                    "PRESENT",
                    "PRESENT",
                    "ABSENT",
                    "ABSENT",
                    "PRESENT",
                ],
            }
        )

        report_ids, features, targets = build_supervised_report_dataset(
            labels,
            "Target",
            ["Signal"],
        )

        self.assertNotIn("r3", report_ids)
        self.assertEqual(list(features.columns), [
            "Signal — present",
            "Signal — absent",
            "Signal — not_mentioned",
        ])
        self.assertEqual(len(targets), 6)
        self.assertEqual(int(targets.sum()), 3)

    def test_cart_and_ensembles_evaluate_binary_findings_deterministically(self):
        labels = pd.DataFrame(
            {
                "report_id": [f"r{i}" for i in range(12)],
                "Target": [
                    "PRESENT",
                    "ABSENT",
                    "PRESENT",
                    "ABSENT",
                    "PRESENT",
                    "ABSENT",
                    "PRESENT",
                    "ABSENT",
                    "PRESENT",
                    "ABSENT",
                    "PRESENT",
                    "ABSENT",
                ],
                "Signal": [
                    "PRESENT",
                    "ABSENT",
                    "PRESENT",
                    "ABSENT",
                    "PRESENT",
                    "ABSENT",
                    "PRESENT",
                    "ABSENT",
                    "PRESENT",
                    "ABSENT",
                    "PRESENT",
                    "ABSENT",
                ],
            }
        )
        _, features, targets = build_supervised_report_dataset(
            labels,
            "Target",
            ["Signal"],
        )

        for model_name in ("CART", "Random forest", "AdaBoost"):
            result = evaluate_binary_classifier(
                features.to_numpy(),
                targets,
                list(features.columns),
                model_name,
                n_estimators=5,
                random_state=9,
            )
            repeated = evaluate_binary_classifier(
                features.to_numpy(),
                targets,
                list(features.columns),
                model_name,
                n_estimators=5,
                random_state=9,
            )

            self.assertEqual(len(result["y_test"]), 2)
            self.assertEqual(result["confusion_matrix"].shape, (2, 2))
            self.assertEqual(result["metrics"], repeated["metrics"])
            self.assertTrue(
                all(0.0 <= value <= 1.0 for value in result["metrics"].values())
            )
            if model_name == "CART":
                self.assertIn("Signal", result["tree_rules"])
                self.assertIn("digraph CART", result["tree_graph"])
                self.assertIn("Predict PRESENT", result["tree_graph"])

    def test_supervised_classifier_rejects_missing_binary_class(self):
        with self.assertRaisesRegex(ValueError, "PRESENT and two ABSENT"):
            evaluate_binary_classifier(
                np.array([[0.0], [1.0], [0.0]]),
                np.array([0, 0, 1]),
                ["Finding"],
                "CART",
            )


if __name__ == "__main__":
    unittest.main()
