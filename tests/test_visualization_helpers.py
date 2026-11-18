import unittest
import datetime
import numpy as np
import pandas as pd
import altair as alt
from src.analytics.visualization_helpers import (
    create_concordance_chart,
    create_concordance_donut,
    create_confidence_ranking_chart,
    create_nlp_breakdown_chart,
    create_timeline_gantt_chart,
    create_stg_rules_chart,
    create_hmm_trajectory_chart,
    create_factor_graph_flow_chart,
    create_confusion_matrix_heatmap,
    create_cohort_prevalence_chart,
    create_cooccurrence_heatmap,
)


class TestVisualizationHelpers(unittest.TestCase):
    def setUp(self):
        self.sample_correlation = [
            {
                "condition": "Pneumonia",
                "image_score": 0.85,
                "image_status": "PRESENT",
                "report_status": "PRESENT",
                "correlation_result": "CONSISTENT"
            },
            {
                "condition": "Pleural Effusion",
                "image_score": 0.12,
                "image_status": "NOT_DETECTED",
                "report_status": "ABSENT",
                "correlation_result": "CONSISTENT"
            },
            {
                "condition": "Edema",
                "image_score": 0.74,
                "image_status": "PRESENT",
                "report_status": "ABSENT",
                "correlation_result": "POTENTIAL_INCONSISTENCY"
            }
        ]
        self.sample_predictions = {
            "Pneumonia": 0.85,
            "Edema": 0.12,
            "Pleural Effusion": 0.05,
            "Pneumothorax": 0.02,
            "Atelectasis": 0.45,
            "Cardiomegaly": 0.18
        }
        self.sample_report = {
            "Pneumonia": "PRESENT",
            "Edema": "ABSENT",
            "Pleural Effusion": "NOT_MENTIONED"
        }

    def test_concordance_chart(self):
        chart = create_concordance_chart(self.sample_correlation)
        d = chart.to_dict()
        self.assertIn("layer", d)

    def test_concordance_donut(self):
        chart = create_concordance_donut(self.sample_correlation)
        d = chart.to_dict()
        self.assertIn("mark", d)

    def test_confidence_ranking_chart(self):
        chart = create_confidence_ranking_chart(self.sample_predictions, "Pneumonia")
        d = chart.to_dict()
        self.assertIn("layer", d)

    def test_nlp_breakdown_chart(self):
        chart = create_nlp_breakdown_chart(self.sample_report)
        d = chart.to_dict()
        self.assertIn("mark", d)

    def test_timeline_gantt_chart(self):
        today = datetime.date.today()
        chart = create_timeline_gantt_chart(today, today + datetime.timedelta(days=3), {})
        d = chart.to_dict()
        self.assertIn("mark", d)

    def test_stg_rules_chart(self):
        stg_res = {"rule_results": ["Image Evidence: Supported", "Timeline: Pass"]}
        chart = create_stg_rules_chart(stg_res)
        d = chart.to_dict()
        self.assertIn("mark", d)

    def test_hmm_trajectory_chart(self):
        df = pd.DataFrame({
            "Study order": [1, 2],
            "Observation": ["ABSENT", "PRESENT"],
            "P(condition present)": [0.1, 0.9],
            "P(condition absent)": [0.9, 0.1]
        })
        chart = create_hmm_trajectory_chart(df, "Pneumonia")
        d = chart.to_dict()
        self.assertIn("layer", d)

    def test_factor_graph_flow_chart(self):
        chart = create_factor_graph_flow_chart(0.2, 0.8, "PRESENT", np.array([0.15, 0.85]))
        d = chart.to_dict()
        self.assertIn("layer", d)

    def test_confusion_matrix_heatmap(self):
        cm = np.array([[50, 5], [10, 35]])
        chart = create_confusion_matrix_heatmap(cm)
        d = chart.to_dict()
        self.assertIn("layer", d)

    def test_cohort_prevalence_chart(self):
        df = pd.DataFrame({
            "Pneumonia": ["PRESENT", "ABSENT", "PRESENT"],
            "Edema": ["ABSENT", "ABSENT", "NOT_MENTIONED"]
        })
        chart = create_cohort_prevalence_chart(df, ["Pneumonia", "Edema"])
        d = chart.to_dict()
        self.assertIn("layer", d)

    def test_cooccurrence_heatmap(self):
        df = pd.DataFrame({
            "Pneumonia": ["PRESENT", "ABSENT", "PRESENT"],
            "Edema": ["PRESENT", "ABSENT", "NOT_MENTIONED"]
        })
        chart = create_cooccurrence_heatmap(df, ["Pneumonia", "Edema"])
        d = chart.to_dict()
        self.assertIn("layer", d)


if __name__ == "__main__":
    unittest.main()
