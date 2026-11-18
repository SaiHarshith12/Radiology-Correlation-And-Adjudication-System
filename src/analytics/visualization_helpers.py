"""
Visualization Helpers for Radiology Correlation System.
Generates responsive, high-clarity Altair charts for clinical decision support.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta


# Clinical Color Palette
COLOR_PASS = "#059669"      # Emerald Green
COLOR_FAIL = "#dc2626"      # Crimson Red
COLOR_WARN = "#d97706"      # Amber Yellow
COLOR_INFO = "#0284c7"      # Sky Blue
COLOR_NEUTRAL = "#64748b"   # Slate Gray
COLOR_PRIMARY = "#2563eb"   # Royal Blue


def create_concordance_chart(correlation_results: list[dict]) -> alt.Chart:
    """
    Horizontal comparative bar chart of Image AI probability vs 0.50 threshold,
    color-coded by multi-modal correlation status.
    """
    rows = []
    for r in correlation_results:
        res = r.get("correlation_result", "UNKNOWN")
        if "CONSISTENT" in res:
            status_group = "Consistent"
        elif "INCONSISTENCY" in res or "MISMATCH" in res:
            status_group = "Potential Inconsistency"
        elif "NEEDS_REVIEW" in res:
            status_group = "Needs Review"
        else:
            status_group = "No Evidence / Absent"

        rows.append({
            "Condition": r["condition"],
            "Image Score": round(float(r["image_score"]), 3),
            "Image Status": r.get("image_status", "N/A"),
            "Report Status": r.get("report_status", "N/A"),
            "Correlation": res.replace("_", " ").title(),
            "Status Group": status_group
        })

    df = pd.DataFrame(rows)

    color_scale = alt.Scale(
        domain=["Consistent", "Needs Review", "Potential Inconsistency", "No Evidence / Absent"],
        range=[COLOR_PASS, COLOR_WARN, COLOR_FAIL, COLOR_NEUTRAL]
    )

    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=22).encode(
        x=alt.X(
            "Image Score:Q",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format="%", title="AI Model Probability")
        ),
        y=alt.Y(
            "Condition:N",
            sort="-x",
            axis=alt.Axis(title="")
        ),
        color=alt.Color("Status Group:N", scale=color_scale, legend=alt.Legend(title="Concordance")),
        tooltip=[
            alt.Tooltip("Condition:N"),
            alt.Tooltip("Image Score:Q", format=".1%"),
            alt.Tooltip("Image Status:N"),
            alt.Tooltip("Report Status:N"),
            alt.Tooltip("Correlation:N")
        ]
    )

    # Threshold rule at 50%
    rule_df = pd.DataFrame([{"threshold": 0.50, "Label": "Diagnostic Cutoff (50%)"}])
    rule = alt.Chart(rule_df).mark_rule(
        color="#b91c1c",
        strokeDash=[5, 5],
        strokeWidth=2
    ).encode(
        x="threshold:Q"
    )

    text = bars.mark_text(
        align="left",
        baseline="middle",
        dx=5,
        color="#1e293b",
        fontWeight=600,
        fontSize=11
    ).encode(
        text=alt.Text("Image Score:Q", format=".1%")
    )

    chart = (bars + rule + text).properties(
        height=260,
        title="Image AI Probability vs. Report Concordance (Threshold: 50%)"
    ).configure_view(strokeWidth=0)

    return chart


def create_concordance_donut(correlation_results: list[dict]) -> alt.Chart:
    """Donut chart showing proportion of consistent vs inconsistent findings."""
    counts = {}
    for r in correlation_results:
        res = r.get("correlation_result", "UNKNOWN")
        if "CONSISTENT" in res:
            cat = "Consistent"
        elif "INCONSISTENCY" in res or "MISMATCH" in res:
            cat = "Inconsistent / Conflict"
        elif "NEEDS_REVIEW" in res:
            cat = "Needs Clinical Review"
        else:
            cat = "Negative / Unmentioned"
        counts[cat] = counts.get(cat, 0) + 1

    df = pd.DataFrame([{"Category": k, "Count": v} for k, v in counts.items()])

    color_scale = alt.Scale(
        domain=["Consistent", "Needs Clinical Review", "Inconsistent / Conflict", "Negative / Unmentioned"],
        range=[COLOR_PASS, COLOR_WARN, COLOR_FAIL, COLOR_NEUTRAL]
    )

    chart = alt.Chart(df).mark_arc(innerRadius=45, outerRadius=75).encode(
        theta=alt.Theta(field="Count", type="quantitative"),
        color=alt.Color(field="Category", type="nominal", scale=color_scale, legend=alt.Legend(title="Agreement")),
        tooltip=[alt.Tooltip("Category:N"), alt.Tooltip("Count:Q")]
    ).properties(
        width=220,
        height=180,
        title="Agreement Breakdown"
    )

    return chart


def create_confidence_ranking_chart(image_predictions: dict[str, float], claimed_condition: str) -> alt.Chart:
    """Ranked horizontal bar chart of image pathologies with claimed condition highlight."""
    rows = []
    for cond, score in image_predictions.items():
        s = float(score)
        is_claimed = (cond.lower() == claimed_condition.lower())
        tier = "High (>80%)" if s >= 0.8 else ("Elevated (50-80%)" if s >= 0.5 else ("Borderline (30-50%)" if s >= 0.3 else "Low (<30%)"))
        rows.append({
            "Condition": cond + (" ★ (Claimed)" if is_claimed else ""),
            "Score": round(s, 3),
            "Tier": tier,
            "IsClaimed": "Claimed Condition" if is_claimed else "Other Finding"
        })

    df = pd.DataFrame(rows).sort_values("Score", ascending=False)

    color_scale = alt.Scale(
        domain=["Claimed Condition", "Other Finding"],
        range=["#ea580c", "#3b82f6"]
    )

    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=5, height=22).encode(
        x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title="Model Probability")),
        y=alt.Y("Condition:N", sort="-x", axis=alt.Axis(title="")),
        color=alt.Color("IsClaimed:N", scale=color_scale, legend=alt.Legend(title="Focus")),
        tooltip=[alt.Tooltip("Condition:N"), alt.Tooltip("Score:Q", format=".1%"), alt.Tooltip("Tier:N")]
    )

    text = bars.mark_text(
        align="left",
        baseline="middle",
        dx=5,
        fontSize=11,
        fontWeight=600
    ).encode(
        text=alt.Text("Score:Q", format=".1%")
    )

    return (bars + text).properties(
        height=260,
        title="Deep Learning Pathology Probabilities (TorchXRayVision DenseNet-121)"
    ).configure_view(strokeWidth=0)


def create_nlp_breakdown_chart(report_predictions: dict[str, str]) -> alt.Chart:
    """Donut and categorical chart of findings extracted from report."""
    rows = []
    for cond, status in report_predictions.items():
        rows.append({
            "Condition": cond,
            "Status": status.replace("_", " ").title()
        })
    df = pd.DataFrame(rows)

    color_scale = alt.Scale(
        domain=["Present", "Absent", "Not Mentioned"],
        range=[COLOR_PASS, COLOR_INFO, COLOR_NEUTRAL]
    )

    chart = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=22).encode(
        x=alt.X("count():Q", axis=alt.Axis(title="Count of Findings", tickMinStep=1)),
        y=alt.Y("Status:N", title="Report Status", sort="-x"),
        color=alt.Color("Status:N", scale=color_scale, legend=None),
        tooltip=[alt.Tooltip("Status:N"), alt.Tooltip("count():Q", title="Findings")]
    ).properties(
        height=180,
        title="Linguistic Finding Status Distribution"
    ).configure_view(strokeWidth=0)

    return chart


def create_timeline_gantt_chart(report_date: datetime.date, claim_date: datetime.date, timeline_result: dict) -> alt.Chart:
    """
    Visual chronological timeline chart showing Exam date, Claim filing date,
    elapsed lag, and guideline window.
    """
    delta_days = (claim_date - report_date).days
    is_valid = (delta_days >= 0 and delta_days <= 60)
    status_label = "Within Standard Window" if is_valid else ("Filing Lag Exceeded (>60d)" if delta_days > 60 else "Anomalous Date (Claim before Exam)")
    bar_color = COLOR_PASS if is_valid else COLOR_FAIL

    events = [
        {
            "Event": "1. Radiologic Exam",
            "Date": str(report_date),
            "Type": "Clinical Reference",
            "Order": 1
        },
        {
            "Event": "2. Claim Submission",
            "Date": str(claim_date),
            "Type": "Administrative Filing",
            "Order": 2
        }
    ]
    df_events = pd.DataFrame(events)

    chart = alt.Chart(df_events).mark_circle(size=250).encode(
        x=alt.X("Date:T", axis=alt.Axis(title="Timeline (Date)")),
        y=alt.Y("Event:N", sort=["1. Radiologic Exam", "2. Claim Submission"], axis=alt.Axis(title="")),
        color=alt.Color("Type:N", scale=alt.Scale(
            domain=["Clinical Reference", "Administrative Filing"],
            range=[COLOR_PRIMARY, bar_color]
        )),
        tooltip=[alt.Tooltip("Event:N"), alt.Tooltip("Date:T", format="%Y-%m-%d"), alt.Tooltip("Type:N")]
    ).properties(
        height=140,
        title=f"Adjudication Timeline Sequence · Elapsed Lag: {delta_days} Days ({status_label})"
    )

    return chart


def create_stg_rules_chart(stg_result: dict) -> alt.Chart:
    """Bar chart summarizing STG guideline rules and verification outcomes."""
    rules = stg_result.get("rule_results", [])
    rows = []
    for idx, rule in enumerate(rules, 1):
        r_lower = rule.lower()
        if "pass" in r_lower or "supported" in r_lower or "met" in r_lower or "consistent" in r_lower or "verified" in r_lower:
            outcome = "PASS"
            score = 1.0
        elif "inconsistent" in r_lower or "fail" in r_lower or "anomaly" in r_lower or "mismatch" in r_lower:
            outcome = "FLAG / FAIL"
            score = 0.2
        else:
            outcome = "REVIEW"
            score = 0.6

        # Short summary label
        label = rule.split(":")[0] if ":" in rule else f"Rule {idx}"
        rows.append({
            "Rule": label[:45] + ("..." if len(label) > 45 else ""),
            "Full Description": rule,
            "Outcome": outcome,
            "Compliance Score": score
        })

    if not rows:
        rows = [{"Rule": "Standard Evidence Review", "Full Description": "No specific rules triggered", "Outcome": "PASS", "Compliance Score": 1.0}]

    df = pd.DataFrame(rows)

    color_scale = alt.Scale(
        domain=["PASS", "REVIEW", "FLAG / FAIL"],
        range=[COLOR_PASS, COLOR_WARN, COLOR_FAIL]
    )

    chart = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=20).encode(
        x=alt.X("Compliance Score:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title="Rule Alignment")),
        y=alt.Y("Rule:N", sort=None, axis=alt.Axis(title="")),
        color=alt.Color("Outcome:N", scale=color_scale, legend=alt.Legend(title="Verification")),
        tooltip=[alt.Tooltip("Rule:N"), alt.Tooltip("Outcome:N"), alt.Tooltip("Full Description:N")]
    ).properties(
        height=max(140, len(rows) * 35),
        title="Statutory Clinical Guideline (STG) Rule Evaluations"
    ).configure_view(strokeWidth=0)

    return chart


def create_hmm_trajectory_chart(hmm_results: pd.DataFrame, condition: str) -> alt.Chart:
    """Interactive longitudinal trajectory chart for HMM state probabilities."""
    melted = hmm_results.melt(
        id_vars=["Study order", "Observation"],
        value_vars=["P(condition present)", "P(condition absent)"],
        var_name="State",
        value_name="Probability"
    )

    color_scale = alt.Scale(
        domain=["P(condition present)", "P(condition absent)"],
        range=[COLOR_FAIL, COLOR_PASS]
    )

    line = alt.Chart(melted).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X("Study order:O", title="Sequential Study Visit (Earliest ➔ Latest)"),
        y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title="Posterior Marginal")),
        color=alt.Color("State:N", scale=color_scale, legend=alt.Legend(title="Latent State")),
        tooltip=[
            alt.Tooltip("Study order:O"),
            alt.Tooltip("Observation:N"),
            alt.Tooltip("State:N"),
            alt.Tooltip("Probability:Q", format=".1%")
        ]
    )

    # 50% cutoff line
    rule_df = pd.DataFrame([{"threshold": 0.50}])
    cutoff = alt.Chart(rule_df).mark_rule(color="#94a3b8", strokeDash=[4, 4]).encode(y="threshold:Q")

    return (line + cutoff).properties(
        height=280,
        title=f"Longitudinal Latent State Progression for '{condition}'"
    )


def create_factor_graph_flow_chart(prior: float, image_score: float, report_status: str, posterior: np.ndarray) -> alt.Chart:
    """Waterfall belief-propagation flow chart for binary sum-product marginalization."""
    image_effect = image_score - prior
    rep_eff = 0.20 if report_status == "PRESENT" else (-0.20 if report_status == "ABSENT" else 0.0)

    steps = [
        {"Stage": "1. Prior Belief", "Value": prior, "Type": "Baseline"},
        {"Stage": "2. Image AI Likelihood", "Value": image_score, "Type": "Evidence"},
        {"Stage": "3. Final Posterior Marginal", "Value": float(posterior[1]), "Type": "Marginal"}
    ]
    df = pd.DataFrame(steps)

    chart = alt.Chart(df).mark_bar(cornerRadiusEnd=5, width=45).encode(
        x=alt.X("Stage:N", sort=None, axis=alt.Axis(title="")),
        y=alt.Y("Value:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title="Probability of Condition Present")),
        color=alt.Color("Type:N", scale=alt.Scale(
            domain=["Baseline", "Evidence", "Marginal"],
            range=[COLOR_NEUTRAL, COLOR_INFO, COLOR_PASS]
        ), legend=None),
        tooltip=[alt.Tooltip("Stage:N"), alt.Tooltip("Value:Q", format=".1%")]
    )

    text = chart.mark_text(dy=-10, fontWeight=600, fontSize=12).encode(
        text=alt.Text("Value:Q", format=".1%")
    )

    return (chart + text).properties(
        height=260,
        title="Sum-Product Factor Graph: Probability Updating Workflow"
    ).configure_view(strokeWidth=0)


def create_confusion_matrix_heatmap(confusion: np.ndarray) -> alt.Chart:
    """Heatmap visualization of 2x2 confusion matrix."""
    total = np.sum(confusion) or 1
    data = [
        {"Actual": "Actual Absent (0)", "Predicted": "Pred Absent (0)", "Count": int(confusion[0, 0]), "Rate": confusion[0, 0] / total},
        {"Actual": "Actual Absent (0)", "Predicted": "Pred Present (1)", "Count": int(confusion[0, 1]), "Rate": confusion[0, 1] / total},
        {"Actual": "Actual Present (1)", "Predicted": "Pred Absent (0)", "Count": int(confusion[1, 0]), "Rate": confusion[1, 0] / total},
        {"Actual": "Actual Present (1)", "Predicted": "Pred Present (1)", "Count": int(confusion[1, 1]), "Rate": confusion[1, 1] / total},
    ]
    df = pd.DataFrame(data)

    rect = alt.Chart(df).mark_rect().encode(
        x=alt.X("Predicted:N", title="Predicted Label"),
        y=alt.Y("Actual:N", title="True Ground Truth"),
        color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), legend=alt.Legend(title="Count")),
        tooltip=[alt.Tooltip("Actual:N"), alt.Tooltip("Predicted:N"), alt.Tooltip("Count:Q"), alt.Tooltip("Rate:Q", format=".1%")]
    )

    text = rect.mark_text(fontSize=14, fontWeight=700).encode(
        text=alt.Text("Count:Q"),
        color=alt.condition(alt.datum.Count > (total / 3), alt.value("white"), alt.value("black"))
    )

    return (rect + text).properties(
        width=300,
        height=220,
        title="Held-Out Validation Confusion Matrix"
    )


def create_cohort_prevalence_chart(cohort_df: pd.DataFrame, conditions: list[str]) -> alt.Chart:
    """Prevalence bar chart across all reports in the cohort."""
    total_reports = len(cohort_df)
    rows = []
    for cond in conditions:
        if cond in cohort_df.columns:
            counts = (cohort_df[cond].str.upper() == "PRESENT").sum()
            rate = counts / total_reports if total_reports > 0 else 0
            rows.append({
                "Condition": cond,
                "Present Cases": int(counts),
                "Prevalence Rate": round(rate, 4)
            })

    df = pd.DataFrame(rows).sort_values("Prevalence Rate", ascending=False)

    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=5, height=22).encode(
        x=alt.X("Prevalence Rate:Q", scale=alt.Scale(domain=[0, max(0.5, df["Prevalence Rate"].max() + 0.05)]), axis=alt.Axis(format="%", title="Prevalence across 7,471 Reports")),
        y=alt.Y("Condition:N", sort="-x", axis=alt.Axis(title="")),
        color=alt.Color("Prevalence Rate:Q", scale=alt.Scale(scheme="tealblues"), legend=None),
        tooltip=[alt.Tooltip("Condition:N"), alt.Tooltip("Present Cases:Q"), alt.Tooltip("Prevalence Rate:Q", format=".2%")]
    )

    text = bars.mark_text(align="left", baseline="middle", dx=5, fontWeight=600, fontSize=11).encode(
        text=alt.Text("Prevalence Rate:Q", format=".1%")
    )

    return (bars + text).properties(
        height=240,
        title=f"Hospital Cohort Pathology Prevalence (Total Studies: {total_reports:,})"
    ).configure_view(strokeWidth=0)


def create_cooccurrence_heatmap(cohort_df: pd.DataFrame, conditions: list[str]) -> alt.Chart:
    """Co-occurrence matrix heatmap among pathologies in the cohort."""
    rows = []
    n = len(cohort_df)
    if n == 0:
        return alt.Chart(pd.DataFrame()).mark_text()

    for c1 in conditions:
        for c2 in conditions:
            if c1 in cohort_df.columns and c2 in cohort_df.columns:
                p1 = (cohort_df[c1].str.upper() == "PRESENT")
                p2 = (cohort_df[c2].str.upper() == "PRESENT")
                both = int((p1 & p2).sum())
                rows.append({
                    "Condition A": c1,
                    "Condition B": c2,
                    "Co-occurrence Count": both
                })

    df = pd.DataFrame(rows)

    rect = alt.Chart(df).mark_rect().encode(
        x=alt.X("Condition B:N", title=""),
        y=alt.Y("Condition A:N", title=""),
        color=alt.Color("Co-occurrence Count:Q", scale=alt.Scale(scheme="purples"), legend=alt.Legend(title="Co-occurs")),
        tooltip=[alt.Tooltip("Condition A:N"), alt.Tooltip("Condition B:N"), alt.Tooltip("Co-occurrence Count:Q")]
    )

    text = rect.mark_text(fontSize=11, fontWeight=600).encode(
        text=alt.Text("Co-occurrence Count:Q"),
        color=alt.condition(alt.datum["Co-occurrence Count"] > 100, alt.value("white"), alt.value("black"))
    )

    return (rect + text).properties(
        height=280,
        title="Pathology Co-occurrence Heatmap across Cohort"
    )
