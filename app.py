import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import json
from datetime import date, datetime, timedelta
from PIL import Image, ImageOps, ImageEnhance

# ==============================================================================
# 1. ADD SRC TO PYTHON PATH
# ==============================================================================

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# ==============================================================================
# 2. IMPORT PROJECT MODULES
# ==============================================================================

IMPORT_ERROR = None

try:
    from image_model.image_predictor import predict_image, get_model_status
    from nlp.condition_extractor import extract_conditions
    from correlation.correlation_engine import correlate_all
    from claim.claim_verifier import verify_claim
    from severity.severity_verifier import verify_severity
    from timeline.timeline_verifier import verify_timeline
    from stg.stg_engine import check_evidence_rules
    from adjudication.adjudication_engine import generate_adjudication_decision
    from explanation.adjudicator_explainer import generate_explanation
    from analytics.advanced_algorithms import (
        REPORT_STATES,
        binary_sum_product,
        build_report_feature_matrix,
        build_supervised_report_dataset,
        evaluate_binary_classifier,
        hmm_forward_backward,
        kmeans,
        pca,
    )
    from analytics.visualization_helpers import (
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
except Exception as e:
    IMPORT_ERROR = str(e)


# ==============================================================================
# 3. PAGE CONFIGURATION & METADATA
# ==============================================================================

st.set_page_config(
    page_title="Radiology Correlation & Adjudication System",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# 4. DATA CACHING
# ==============================================================================

@st.cache_data(max_entries=2)
def load_report_labels(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


@st.cache_data(max_entries=4)
def run_cohort_kmeans(values: np.ndarray, n_clusters: int):
    return kmeans(values, n_clusters, random_state=42)


CLASSIFIER_RESULT_VERSION = 2

@st.cache_data(max_entries=8)
def run_cohort_classifier(
    values,
    targets,
    feature_names,
    model_name,
    test_fraction,
    max_depth,
    n_estimators,
    result_version,
):
    results = evaluate_binary_classifier(
        values,
        targets,
        feature_names,
        model_name,
        test_fraction=test_fraction,
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=42,
    )
    results["result_version"] = result_version
    return results


# ==============================================================================
# 5. CLINICAL SCENARIOS / PRESET CASES
# ==============================================================================

SAMPLE_CASES = {
    "Case 1: Confirmed Pneumonia (High Concordance)": {
        "condition": "Pneumonia",
        "severity": "Moderate",
        "report_days_ago": 3,
        "claim_days_ago": 1,
        "image_file": "CXR1000_IM-0003-1001.png",
        "report_text": (
            "There is increased opacity within the right upper lobe with focal consolidation "
            "and associated area of atelectasis. The cardiac silhouette is within normal limits. "
            "No pleural effusion or pneumothorax identified. Findings are consistent with bacterial pneumonia."
        ),
        "tag": "EVIDENCE CONSISTENT",
        "description": "True Positive: Clinical findings and radiologic consolidation agree with the claimed pneumonia."
    },
    "Case 2: Claim Discrepancy / Mismatch (Fraud Alert)": {
        "condition": "Pneumonia",
        "severity": "Severe",
        "report_days_ago": 5,
        "claim_days_ago": 2,
        "image_file": "CXR1_1_IM-0001-3001.png",
        "report_text": (
            "The cardiac silhouette and mediastinum size are within normal limits. "
            "There is no pulmonary edema. There is no focal consolidation or acute pneumonia. "
            "There are no signs of a pleural effusion. There is no evidence of pneumothorax. Normal chest x-ray."
        ),
        "tag": "FLAGGED: CLAIM MISMATCH",
        "description": "Conflict: Claim asserts Severe Pneumonia, but radiologist explicitly notes 'no focal consolidation' and AI detects clear lungs."
    },
    "Case 3: Pleural Effusion with Timeline Anomaly": {
        "condition": "Pleural Effusion",
        "severity": "Moderate",
        "report_days_ago": 2,
        "claim_days_ago": 15,  # Claim date earlier than imaging exam!
        "image_file": "CXR10_IM-0002-1001.png",
        "report_text": (
            "The cardiomediastinal silhouette is within normal limits. "
            "Blunting of the right costophrenic angle with small-to-moderate right pleural effusion. "
            "No focal consolidation. No pneumothorax."
        ),
        "tag": "FLAGGED: TIMELINE ANOMALY",
        "description": "Procedural Anomaly: The claim date precedes the date of radiological examination."
    },
    "Case 4: Cardiomegaly with Severity Discordance": {
        "condition": "Cardiomegaly",
        "severity": "Severe",
        "report_days_ago": 7,
        "claim_days_ago": 3,
        "image_file": "CXR1004_IM-0005-1001.png",
        "report_text": (
            "Mild cardiac enlargement. Cardiothoracic ratio is borderline at approximately 0.52. "
            "Pulmonary vascularity is within normal limits without evidence of acute congestive failure or edema."
        ),
        "tag": "FLAGGED: SEVERITY DISCORDANCE",
        "description": "Severity Discordance: Claimed 'Severe' cardiomegaly, but objective findings show mild borderline prominence only."
    },
    "Case 5: Baseline Normal Bilateral Study": {
        "condition": "Pneumothorax",
        "severity": "Mild",
        "report_days_ago": 4,
        "claim_days_ago": 2,
        "image_file": "CXR100_IM-0002-1001.png",
        "report_text": (
            "Both lungs are clear and expanded. Heart and mediastinum normal. "
            "Pleural spaces are free of fluid or pneumothorax. No active cardiopulmonary process."
        ),
        "tag": "NEGATIVE BASELINE",
        "description": "Negative Baseline: Clear bilateral study with no acute pathologic findings."
    },
    "Custom Upload": {
        "condition": "Pneumonia",
        "severity": "Moderate",
        "report_days_ago": 0,
        "claim_days_ago": 0,
        "image_file": None,
        "report_text": "",
        "tag": "USER INPUT",
        "description": "Upload any custom chest X-ray and type/paste custom radiology report text."
    }
}


# ==============================================================================
# 6. ENHANCED MEDICAL CSS STYLES
# ==============================================================================

st.markdown(
    """
    <style>
        /* Base surface & typography */
        .main {
            background-color: #f8fafc;
            color: #0f172a;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
            max-width: 1350px;
        }

        /* Header typography */
        .app-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border-radius: 12px;
            padding: 1.25rem 1.75rem;
            color: #ffffff;
            margin-bottom: 1.2rem;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.15);
        }
        .app-title {
            font-size: 1.65rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin: 0;
            color: #ffffff;
        }
        .app-subtitle {
            font-size: 0.9rem;
            color: #94a3b8;
            margin-top: 0.25rem;
            margin-bottom: 0;
        }

        /* Stepper Pipeline */
        .stepper-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 0.75rem 1.25rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        }
        .step-item {
            display: flex;
            align-items: center;
            font-size: 0.85rem;
            font-weight: 600;
            color: #475569;
        }
        .step-bubble {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: #e2e8f0;
            color: #475569;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 0.5rem;
            font-size: 0.75rem;
        }
        .step-bubble.active {
            background: #2563eb;
            color: #ffffff;
        }
        .step-bubble.completed {
            background: #059669;
            color: #ffffff;
        }
        .step-connector {
            flex: 1;
            height: 2px;
            background: #e2e8f0;
            margin: 0 0.8rem;
        }

        /* Section Header */
        .section-header {
            font-size: 1.1rem;
            font-weight: 700;
            color: #0f172a;
            margin-top: 0.3rem;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Clean Card Surfaces */
        .med-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 1.1rem 1.3rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
            margin-bottom: 1rem;
        }
        .med-hero-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-left: 6px solid #2563eb;
            border-radius: 10px;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.06);
            margin-bottom: 1.2rem;
        }
        .med-hero-card.flagged {
            border-left-color: #dc2626;
            background: #fff8f8;
        }
        .med-hero-card.consistent {
            border-left-color: #059669;
            background: #f8fdfa;
        }

        /* Badges */
        .badge {
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.82rem;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }
        .badge-red {
            background-color: #fee2e2;
            color: #991b1b;
            border: 1px solid #fecaca;
        }
        .badge-green {
            background-color: #d1fae5;
            color: #065f46;
            border: 1px solid #a7f3d0;
        }
        .badge-amber {
            background-color: #fef3c7;
            color: #92400e;
            border: 1px solid #fde68a;
        }
        .badge-blue {
            background-color: #e0f2fe;
            color: #075985;
            border: 1px solid #bae6fd;
        }

        /* KPI Metric Mini Card */
        .metric-box {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            text-align: center;
        }
        .metric-label {
            font-size: 0.75rem;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .metric-val {
            font-size: 1.35rem;
            font-weight: 700;
            color: #0f172a;
            margin-top: 0.2rem;
        }

        /* Image View Container */
        .image-inspect-frame {
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            overflow: hidden;
            background: #000000;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_badge(text: str, level: str = "green") -> str:
    css = {
        "red": "badge-red",
        "green": "badge-green",
        "amber": "badge-amber",
        "blue": "badge-blue"
    }.get(level, "badge-blue")
    return f'<span class="badge {css}">{text}</span>'


# ==============================================================================
# 7. TOP HEADER & PIPELINE STEPPER
# ==============================================================================

st.markdown(
    """
    <div class="app-header">
        <div>
            <div class="app-title">🩻 Radiological Correlation & Claim Adjudication System</div>
            <div class="app-subtitle">
                AI-Assisted Cross-Modal Correlation between Chest Radiography & Radiology Reports for Adjudication Support
            </div>
        </div>
        <div>
            <span class="badge badge-blue">Clinical AI DSS · v2.5</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if IMPORT_ERROR:
    st.error(
        f"⚠️ Dependency import error detected: `{IMPORT_ERROR}`. The dashboard UI will still render."
    )

# Session State Initialization
if "has_results" not in st.session_state:
    st.session_state.has_results = False
if "active_case" not in st.session_state:
    st.session_state.active_case = "Case 1: Confirmed Pneumonia (High Concordance)"
if "contrast_mode" not in st.session_state:
    st.session_state.contrast_mode = "Standard Radiograph"


# Stepper visual showing pipeline progress
s1 = "completed" if st.session_state.has_results else "active"
s2 = "completed" if st.session_state.has_results else ""
s3 = "completed" if st.session_state.has_results else ""
s4 = "completed" if st.session_state.has_results else ""

st.markdown(
    f"""
    <div class="stepper-container">
        <div class="step-item">
            <div class="step-bubble {s1}">1</div>
            <span>Image AI (DenseNet-121)</span>
        </div>
        <div class="step-connector"></div>
        <div class="step-item">
            <div class="step-bubble {s2}">2</div>
            <span>Report NLP & Negation</span>
        </div>
        <div class="step-connector"></div>
        <div class="step-item">
            <div class="step-bubble {s3}">3</div>
            <span>Correlation Engine</span>
        </div>
        <div class="step-connector"></div>
        <div class="step-item">
            <div class="step-bubble {s4}">4</div>
            <span>Policy & Adjudication</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 8. SIDEBAR CONFIGURATION
# ==============================================================================

with st.sidebar:
    st.markdown("### 📋 Case & Claim Parameters")
    st.caption("Select a preset scenario or configure custom claim criteria.")

    # Clinical Scenario Selector
    selected_scenario = st.selectbox(
        "Clinical Scenario Preset",
        list(SAMPLE_CASES.keys()),
        index=list(SAMPLE_CASES.keys()).index(st.session_state.active_case),
        key="scenario_select"
    )

    preset = SAMPLE_CASES[selected_scenario]
    st.info(f"**{preset['tag']}**\n\n{preset['description']}")

    st.markdown("---")
    st.markdown("#### Claim Attributes")

    claimed_condition = st.selectbox(
        "Claimed Condition",
        [
            "Pneumonia",
            "Edema",
            "Pleural Effusion",
            "Pneumothorax",
            "Atelectasis",
            "Cardiomegaly",
        ],
        index=[
            "Pneumonia",
            "Edema",
            "Pleural Effusion",
            "Pneumothorax",
            "Atelectasis",
            "Cardiomegaly",
        ].index(preset["condition"]) if preset["condition"] in [
            "Pneumonia", "Edema", "Pleural Effusion", "Pneumothorax", "Atelectasis", "Cardiomegaly"
        ] else 0,
        key="claim_cond"
    )

    claimed_severity = st.selectbox(
        "Claimed Severity",
        ["Mild", "Moderate", "Severe"],
        index=["Mild", "Moderate", "Severe"].index(preset["severity"]),
        key="claim_sev"
    )

    # Date calculations based on preset
    today = date.today()
    default_report_date = today - timedelta(days=preset.get("report_days_ago", 3))
    default_claim_date = today - timedelta(days=preset.get("claim_days_ago", 1))

    report_date = st.date_input(
        "Radiology Exam Date",
        value=default_report_date,
        key="rep_date"
    )

    claim_date = st.date_input(
        "Claim Filing Date",
        value=default_claim_date,
        key="clm_date"
    )

    delta_days = (claim_date - report_date).days
    if delta_days < 0:
        st.warning(f"⚠️ Filing date is {abs(delta_days)} days before exam date (Timeline Anomaly).")
    elif delta_days > 60:
        st.warning(f"⚠️ Filing lag of {delta_days} days exceeds standard 60-day policy limit.")
    else:
        st.caption(f"✓ Filing lag: {delta_days} days (within standard window).")

    st.markdown("---")

    # System Status & Diagnostic Metadata
    with st.expander("🛠️ System & Model Engine Status", expanded=False):
        model_name = get_model_status()
        st.write(f"**Vision AI:** {model_name}")
        st.write(f"**NLP Parser:** Lexicon with Contextual Negation Check")
        st.write(f"**IU X-Ray Cohort:** 7,471 Patient Studies Available")

    st.caption("Decision-support system for claims adjudicators. Does not replace primary medical review.")


# ==============================================================================
# 9. MAIN INPUT AREA (IMAGE & REPORT)
# ==============================================================================

input_col1, input_col2 = st.columns([0.45, 0.55])

with input_col1:
    st.markdown('<div class="section-header"><span>🩻</span> 1. Chest Radiograph (X-ray)</div>', unsafe_allow_html=True)

    loaded_image = None
    image_display_path = None

    if selected_scenario != "Custom Upload" and preset["image_file"]:
        preset_img_path = os.path.join("data", "images", preset["image_file"])
        if os.path.exists(preset_img_path):
            image_display_path = preset_img_path
            loaded_image = Image.open(preset_img_path)
            st.caption(f"Preset Image: `{preset['image_file']}` (Auto-Loaded)")
        else:
            st.warning("Preset image file not found in `data/images`.")

    custom_file = st.file_uploader(
        "Or upload custom X-ray (PNG, JPG)",
        type=["png", "jpg", "jpeg"],
        key="custom_uploader"
    )

    if custom_file is not None:
        loaded_image = Image.open(custom_file)
        os.makedirs("temp", exist_ok=True)
        custom_path = os.path.join("temp", custom_file.name)
        with open(custom_path, "wb") as f:
            f.write(custom_file.getbuffer())
        image_display_path = custom_path

    if loaded_image is not None:
        # Radiographic viewing filters
        contrast_choice = st.radio(
            "Viewer Filter Mode",
            ["Standard Radiograph", "Bone Invert (PACS LUT)", "High Contrast Enhancement"],
            horizontal=True,
            key="viewer_contrast"
        )

        display_img = loaded_image
        if contrast_choice == "Bone Invert (PACS LUT)":
            display_img = ImageOps.invert(loaded_image.convert("L"))
        elif contrast_choice == "High Contrast Enhancement":
            display_img = ImageEnhance.Contrast(loaded_image.convert("L")).enhance(1.6)

        st.image(display_img, caption=f"Chest Radiograph — {contrast_choice}", width='stretch')
    else:
        st.info("No chest radiograph loaded. Select a preset case from the sidebar or upload an image.")


with input_col2:
    st.markdown('<div class="section-header"><span>📄</span> 2. Hospital Radiology Report</div>', unsafe_allow_html=True)

    default_text = preset.get("report_text", "")
    report_text = st.text_area(
        "Radiology Report Text",
        value=default_text,
        height=280,
        placeholder="Paste clinical findings and impression sections from the hospital radiology report...",
        key="report_input"
    )

    st.caption("Tip: Include clinical sections like 'Findings' and 'Impression'. Negation phrases ('no evidence of', 'without') are detected.")


# ==============================================================================
# 10. RUN ANALYSIS BUTTON & PIPELINE
# ==============================================================================

st.write("")
col_btn1, col_btn2 = st.columns([0.75, 0.25])

with col_btn1:
    analyze_button = st.button(
        "🔍 Run Multi-Modal Correlation & Adjudicate Case",
        type="primary",
        width='stretch',
        disabled=bool(IMPORT_ERROR),
    )

with col_btn2:
    if st.button("🔄 Reset View", width='stretch'):
        st.session_state.has_results = False
        st.rerun()


if analyze_button:
    if loaded_image is None or image_display_path is None:
        st.error("Please upload or select an X-ray image.")
    elif not report_text.strip():
        st.error("Please provide the radiology report text.")
    else:
        with st.status("Executing Multi-Modal Clinical Adjudication Pipeline...", expanded=True) as status:
            # 1. Vision AI Inference
            st.write("🩻 Running deep learning chest X-ray pathology classifier...")
            image_predictions = predict_image(image_display_path)

            # 2. NLP Extraction
            st.write("📄 Extracting clinical entities and detecting negation boundaries...")
            report_predictions = extract_conditions(report_text)

            # 3. Correlation Engine
            st.write("🔗 Correlating image probabilities with linguistic report findings...")
            correlation_results = correlate_all(image_predictions, report_predictions)

            # 4. Claim Verification
            st.write("✅ Verifying asserted claim against multimodal evidence...")
            claim_result = verify_claim(claimed_condition, correlation_results)

            # 5. Severity Verification
            st.write("📊 Evaluating claimed severity concordance...")
            image_score = claim_result.get("image_score", 0.0)
            severity_result = verify_severity(claimed_severity, image_score)

            # 6. Timeline Verification
            st.write("📅 Auditing submission date timeline & filing window...")
            timeline_result = verify_timeline(str(report_date), str(claim_date))

            # 7. Statutory Guidelines Rules
            st.write("📋 Checking statutory clinical guidelines & policy rules...")
            stg_result = check_evidence_rules(claimed_condition, claim_result, timeline_result)

            # 8. Adjudication Decision
            st.write("🏥 Formulating final adjudication recommendation...")
            adjudication_result = generate_adjudication_decision(
                claim_result,
                severity_result,
                timeline_result,
                stg_result,
            )

            # 9. Adjudicator Explanation
            explanation = generate_explanation(claim_result)

            status.update(
                label="Multi-Modal Adjudication Complete!",
                state="complete",
                expanded=False
            )

        # Cache to session state
        st.session_state.has_results = True
        st.session_state.adjudication_result = adjudication_result
        st.session_state.correlation_results = correlation_results
        st.session_state.image_predictions = image_predictions
        st.session_state.report_predictions = report_predictions
        st.session_state.claim_result = claim_result
        st.session_state.severity_result = severity_result
        st.session_state.timeline_result = timeline_result
        st.session_state.stg_result = stg_result
        st.session_state.explanation = explanation
        st.session_state.claimed_condition = claimed_condition
        st.session_state.claimed_severity = claimed_severity
        st.session_state.report_date = report_date
        st.session_state.claim_date = claim_date
        st.session_state.image_path = image_display_path


# ==============================================================================
# 11. RESULTS & ADJUDICATION DASHBOARD
# ==============================================================================

if st.session_state.has_results:
    adjudication_result = st.session_state.adjudication_result
    correlation_results = st.session_state.correlation_results
    image_predictions = st.session_state.image_predictions
    report_predictions = st.session_state.report_predictions
    claim_result = st.session_state.claim_result
    severity_result = st.session_state.severity_result
    timeline_result = st.session_state.timeline_result
    stg_result = st.session_state.stg_result
    explanation = st.session_state.explanation
    claimed_cond = st.session_state.claimed_condition
    claimed_sev = st.session_state.claimed_severity
    rep_date = st.session_state.report_date
    clm_date = st.session_state.claim_date

    decision = adjudication_result.get("decision", "UNKNOWN")
    risk_level = adjudication_result.get("risk_level", "UNKNOWN")
    reasons = adjudication_result.get("reasons", [])
    is_flagged = (decision == "FLAG_FOR_HUMAN_REVIEW")

    st.markdown("---")

    # Hero Recommendation Banner
    hero_class = "flagged" if is_flagged else "consistent"
    badge_type = "red" if is_flagged else "green"
    badge_txt = "FLAGGED FOR HUMAN REVIEW" if is_flagged else "EVIDENCE CONSISTENT"
    action_headline = (
        "Potential Inconsistency / Discrepancy Detected — Mandatory Adjudicator Review Required"
        if is_flagged
        else "Multimodal Evidence Corroborates Claim — Eligible to Proceed"
    )

    st.markdown(
        f"""
        <div class="med-hero-card {hero_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <div>{render_badge(badge_txt, badge_type)}</div>
                <div style="font-size: 0.85rem; font-weight: 700; color: {'#991b1b' if is_flagged else '#065f46'};">
                    RISK LEVEL: {str(risk_level).upper()}
                </div>
            </div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-bottom: 0.35rem;">
                {action_headline}
            </div>
            <div style="font-size: 0.92rem; color: #475569;">
                {explanation.get("reason", "Evidence evaluated across imaging, text NLP, timeline, and statutory rules.")}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top KPI Metrics Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Claimed Condition</div>
                <div class="metric-val">{claimed_cond}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi2:
        image_score_val = claim_result.get("image_score", 0.0)
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">AI Image Probability</div>
                <div class="metric-val">{image_score_val:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi3:
        rep_status_val = claim_result.get("report_status", "NOT_MENTIONED").replace("_", " ").title()
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Report NLP Finding</div>
                <div class="metric-val">{rep_status_val}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi4:
        lag_val = (clm_date - rep_date).days
        lag_display = f"{lag_val} Days" if lag_val >= 0 else f"{lag_val}d (Anomaly)"
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Timeline Lag</div>
                <div class="metric-val">{lag_display}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Rationale & Recommended Actions
    with st.expander("📌 Adjudicator Decision Rationale & Next Steps", expanded=True):
        col_rat1, col_rat2 = st.columns([0.6, 0.4])
        with col_rat1:
            st.markdown("**Key Finding Triggers:**")
            if reasons:
                for r in reasons:
                    st.markdown(f"- {r}")
            else:
                st.markdown("- All clinical, radiographic, and timeline evidence align with policy requirements.")

        with col_rat2:
            st.markdown("**Actionable Next Steps:**")
            st.info(explanation.get("recommended_action", "Proceed with standard claim settlement protocol."))

    st.write("")


    # ==============================================================================
    # 12. DETAILED EVIDENCE TABS WITH GRAPHS
    # ==============================================================================

    (
        tab_compare,
        tab_image,
        tab_report,
        tab_timeline,
        tab_evidence,
        tab_analytics,
        tab_cohort,
    ) = st.tabs(
        [
            "🔗 Dual Correlation",
            "🩻 Image AI Findings",
            "📄 Report NLP",
            "📅 Timeline & Policy",
            "📋 Guideline Rules",
            "🧠 Advanced Analytics",
            "📊 Cohort Population",
        ]
    )

    # --------------------------------------------------------------------------
    # TAB 1: DUAL CORRELATION
    # --------------------------------------------------------------------------
    with tab_compare:
        st.markdown("#### 🔗 Multimodal Concordance Analysis (Image AI vs. Radiology Report)")
        st.caption(
            "Compares deep learning visual probability against NLP findings extracted from the written report. "
            "Conditions above 50% without confirmation or explicitly negated are flagged."
        )

        # Graph 1 & 2: Concordance Bar Chart & Agreement Donut
        col_g1, col_g2 = st.columns([0.7, 0.3])
        with col_g1:
            concordance_chart = create_concordance_chart(correlation_results)
            st.altair_chart(concordance_chart, width='stretch')
        with col_g2:
            donut_chart = create_concordance_donut(correlation_results)
            st.altair_chart(donut_chart, width='stretch')

        # Interactive Table
        comparison_df = pd.DataFrame(
            [
                {
                    "Condition": r["condition"],
                    "Image AI Score": round(float(r["image_score"]), 3),
                    "Image Finding": r["image_status"],
                    "Report Finding": r["report_status"],
                    "Concordance Status": r["correlation_result"].replace("_", " ").title(),
                }
                for r in correlation_results
            ]
        )

        def style_concordance(v):
            v_lower = str(v).lower()
            if "inconsistency" in v_lower or "mismatch" in v_lower:
                return "background-color: #fee2e2; color: #991b1b; font-weight: 600;"
            elif "consistent" in v_lower:
                return "background-color: #d1fae5; color: #065f46; font-weight: 600;"
            elif "needs review" in v_lower:
                return "background-color: #fef3c7; color: #92400e; font-weight: 600;"
            return "background-color: #f1f5f9; color: #475569;"

        styled_df = comparison_df.style.map(
            style_concordance,
            subset=["Concordance Status"]
        ).format({"Image AI Score": "{:.1%}"})

        st.dataframe(styled_df, width='stretch', hide_index=True)

        with st.expander("📐 Formula & Current Values", expanded=False):
            claimed_row = next(
                (row for row in correlation_results if row["condition"] == claimed_cond),
                None,
            )
            claimed_image_score = (
                float(claimed_row["image_score"]) if claimed_row else 0.0
            )
            st.markdown(
                r"""
**Image finding rule**
$$
I_c = \begin{{cases}}
\text{{PRESENT}} & p_c \geq 0.50 \\
\text{{NOT DETECTED}} & p_c < 0.50
\end{{cases}}
$$

For the claimed condition **{condition}**, $p_c={score:.1%}$, so
**{status}**. Concordance is then classified from $I_c$ and the report
finding (PRESENT, ABSENT, or NOT_MENTIONED).
""".format(
                    condition=claimed_cond,
                    score=claimed_image_score,
                    status="PRESENT" if claimed_image_score >= 0.50 else "NOT DETECTED",
                )
            )

        with st.expander("🔍 Condition-by-Condition Diagnostic Details", expanded=False):
            for res in correlation_results:
                c_name = res["condition"]
                c_scr = float(res["image_score"])
                c_img = res["image_status"]
                c_rep = res["report_status"]
                c_res = res["correlation_result"]

                c1, c2, c3 = st.columns([0.3, 0.35, 0.35])
                c1.write(f"**{c_name}**")
                c2.write(f"Image: `{c_scr:.1%}` ({c_img})")
                c3.write(f"Report: `{c_rep}` ➔ **{c_res}**")

    # --------------------------------------------------------------------------
    # TAB 2: IMAGE AI FINDINGS
    # --------------------------------------------------------------------------
    with tab_image:
        st.markdown("#### 🩻 Chest Radiography Deep Learning Findings")
        st.caption("Pathology probabilities generated by TorchXRayVision DenseNet-121 architecture.")

        col_img_g1, col_img_g2 = st.columns([0.65, 0.35])
        with col_img_g1:
            rank_chart = create_confidence_ranking_chart(image_predictions, claimed_cond)
            st.altair_chart(rank_chart, width='stretch')
        with col_img_g2:
            st.markdown("**Claim Target Verification:**")
            claimed_ai_score = image_predictions.get(claimed_cond, 0.0)
            st.metric(
                label=f"AI Probability: {claimed_cond}",
                value=f"{claimed_ai_score:.1%}",
                delta="Elevated Evidence" if claimed_ai_score >= 0.50 else "Below 50% Threshold",
                delta_color="normal" if claimed_ai_score >= 0.50 else "inverse"
            )
            st.caption("50% is standard detection threshold for acute pulmonary opacities.")
            st.markdown(f"**Engine in use:** `{get_model_status()}`")

            with st.expander("📐 DenseNet Prediction Formula", expanded=False):
                st.latex(
                    r"\hat{y}_c=\sigma(z_c)=\frac{1}{1+e^{-z_c}}"
                )
                st.markdown(
                    f"""
- **z_c**: DenseNet-121 output for condition c
- **hat(y_c)**: probability for condition c
- **Cutoff**: D_c = 1 when y_c >= 0.50
- **Claimed condition**: **{claimed_cond}**
- **Current probability**: **{claimed_ai_score:.1%}**
- **Current decision**: **{"PRESENT" if claimed_ai_score >= 0.50 else "NOT DETECTED"}**
"""
                )

    # --------------------------------------------------------------------------
    # TAB 3: REPORT NLP
    # --------------------------------------------------------------------------
    with tab_report:
        st.markdown("#### 📄 Radiology Report NLP & Negation Analysis")
        st.caption("Condition mentions and contextual negation detection parsed from clinical text.")

        col_nlp1, col_nlp2 = st.columns([0.5, 0.5])
        with col_nlp1:
            nlp_chart = create_nlp_breakdown_chart(report_predictions)
            st.altair_chart(nlp_chart, width='stretch')
        with col_nlp2:
            st.markdown("**Extracted Findings Summary:**")
            for cond, status in report_predictions.items():
                if status == "PRESENT":
                    st.markdown(f"🟢 **{cond}** — PRESENT (Explicitly Described)")
                elif status == "ABSENT":
                    st.markdown(f"🔵 **{cond}** — ABSENT (Negated by Radiologist)")
                else:
                    st.markdown(f"⚪ **{cond}** — Not Mentioned")

        with st.expander("📐 NLP Rule Formula & Current Claim Value", expanded=False):
            claimed_report_status = report_predictions.get(
                claimed_cond,
                "NOT_MENTIONED",
            )
            st.markdown(
                r"""
$$
R_c =
\begin{{cases}}
\text{{PRESENT}} & k_c\text{{ found and no negation in preceding 150 characters}} \\
\text{{ABSENT}} & k_c\text{{ found and a negation phrase is present}} \\
\text{{NOT MENTIONED}} & k_c\text{{ not found}}
\end{{cases}}
$$

- **Claimed condition:** `{condition}`
- **Current report status:** **{status}**
- Lexicon terms are matched case-insensitively.
""".format(
                    condition=claimed_cond,
                    status=claimed_report_status,
                )
            )

        with st.expander("📝 View Full Report Transcript", expanded=False):
            st.text(report_text)

    # --------------------------------------------------------------------------
    # TAB 4: TIMELINE & POLICY
    # --------------------------------------------------------------------------
    with tab_timeline:
        st.markdown("#### 📅 Chronological Timeline & Filing Window Audit")
        st.caption("Verifies temporal sequence between image acquisition and insurance claim filing.")

        timeline_chart = create_timeline_gantt_chart(rep_date, clm_date, timeline_result)
        st.altair_chart(timeline_chart, width='stretch')

        tl_col1, tl_col2 = st.columns(2)
        with tl_col1:
            st.metric(
                "Timeline Status",
                timeline_result.get("timeline_status", "UNKNOWN"),
                delta=f"Lag: {delta_days} days"
            )
        with tl_col2:
            st.markdown(f"**Exam Date:** `{rep_date}`")
            st.markdown(f"**Claim Date:** `{clm_date}`")
            st.write(timeline_result.get("message", "Standard filing sequence confirmed."))

        with st.expander("📐 Timeline Formula & Current Values", expanded=False):
            st.markdown(
                r"""
$$
\Delta t=(\text{{Claim Date}}-\text{{Exam Date}})\text{{ in calendar days}}
$$

| Value | Result |
|---|---:|
| $\Delta t$ | **{lag_val} days** |
| 0–30 days | Consistent |
| Negative days | Potential inconsistency |
| >30 days | Needs review |

**Status:** **{status}**
""".format(
                    lag_val=delta_days,
                    status=timeline_result.get("timeline_status", "UNKNOWN"),
                )
            )

    # --------------------------------------------------------------------------
    # TAB 5: GUIDELINE RULES
    # --------------------------------------------------------------------------
    with tab_evidence:
        st.markdown("#### 📋 Statutory Clinical Guidelines (STG) Rule Engine")
        st.caption("Evaluates institutional rules governing diagnostic necessity, timeline windows, and severity corroboration.")

        rules_chart = create_stg_rules_chart(stg_result)
        st.altair_chart(rules_chart, width='stretch')

        st.markdown(f"**Overall Guideline Status:** {render_badge(stg_result.get('stg_status', 'UNKNOWN'), 'green' if 'met' in str(stg_result.get('stg_status')).lower() or 'consistent' in str(stg_result.get('stg_status')).lower() else 'amber')}", unsafe_allow_html=True)

        with st.expander("📐 Guideline Rule Formula & Current Values", expanded=False):
            st.markdown(
        r"""
**Evidence-consistent condition**
$$
E=\left(p_c\geq0.50\right)
\land(C=\text{{CONSISTENT}})
\land(T=\text{{CONSISTENT}})
$$

- $p_c={image_score:.1%}$
- Correlation: **{correlation}**
- Timeline: **{timeline}**
- Overall STG status: **{status}**
- Rule threshold: **0.50**
""".format(
            image_score=float(claim_result.get("image_score", 0.0)),
            correlation=claim_result.get("correlation_result", "UNKNOWN"),
            timeline=timeline_result.get("timeline_status", "UNKNOWN"),
            status=stg_result.get("stg_status", "UNKNOWN"),
        )
            )

    # --------------------------------------------------------------------------
    # TAB 6: ADVANCED ANALYTICS
    # --------------------------------------------------------------------------
    with tab_analytics:
        st.markdown("#### 🧠 Exploratory Algorithm Lab")
        st.caption(
            "Educational and diagnostic analytical methods (K-Means, PCA, HMM, Factor Graph, Supervised CART/AdaBoost/Random Forest)."
        )

        selected_method = st.segmented_control(
            "Select Analytical Methodology",
            ["K-means & PCA", "Hidden Markov model", "Sum-product algorithm", "Supervised ML"],
            default="K-means & PCA",
            key="adv_algo_choice"
        )

        labels_path = os.path.join(
            os.path.dirname(__file__), "data", "processed", "iu_xray_nlp_labels.csv"
        )

        if selected_method == "K-means & PCA":
            st.markdown("##### Finding Profile Clustering & Dimensionality Reduction")
            try:
                cohort = load_report_labels(labels_path)
                report_ids, feature_frame, feature_matrix = build_report_feature_matrix(
                    cohort, list(image_predictions.keys())
                )
                unique_profiles = len(np.unique(feature_matrix, axis=0))
                max_clusters = min(8, unique_profiles)

                n_clusters = st.slider("Number of Clusters (k)", 2, max_clusters, min(3, max_clusters))
                cluster_labels, centers, inertia = run_cohort_kmeans(feature_matrix, n_clusters)
                projection, explained_variance = pca(feature_matrix, n_components=2)

                plot_data = pd.DataFrame({
                    "Report ID": report_ids,
                    "PC1": projection[:, 0],
                    "PC2": projection[:, 1],
                    "Cluster": [f"Cluster {l + 1}" for l in cluster_labels]
                })

                c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
                c_kpi1.metric("Unique Patient Reports", len(report_ids))
                c_kpi2.metric("Distinct Finding Profiles", unique_profiles)
                c_kpi3.metric("K-Means Inertia", f"{inertia:.2f}")

                st.scatter_chart(plot_data, x="PC1", y="PC2", color="Cluster")
                st.caption(f"Variance Explained: PC1 {explained_variance[0]:.1%}, PC2 {explained_variance[1]:.1%}")

                # Cluster Centroid Profile Heatmap
                centroid_frame = pd.DataFrame(
                    centers,
                    columns=feature_frame.columns,
                    index=[f"Cluster {i + 1}" for i in range(n_clusters)]
                )
                with st.expander("Cluster Centroid Profiles (Average Features)", expanded=False):
                    st.dataframe(centroid_frame)

                with st.expander("📐 K-Means & PCA Formula", expanded=False):
                    st.markdown("**K-Means objective**")
                    st.latex(
                        r"J=\sum_{i=1}^{n}\sum_{k=1}^{K}"
                        r"w_{ik}\lVert x_i-\mu_k\rVert^2"
                    )
                    st.markdown(
                        f"""
- **K:** {n_clusters} clusters
- Unique report profiles: **{unique_profiles}**
- Inertia J: **{inertia:.2f}**
- PC1 variance: **{float(explained_variance[0]):.1%}**
- PC2 variance: **{float(explained_variance[1]):.1%}**

PCA is calculated from the singular value decomposition of the centered feature matrix.
"""
                    )

            except Exception as e:
                st.error(f"Clustering error: {e}")

        elif selected_method == "Hidden Markov model":
            st.markdown("##### Longitudinal Latent Disease Progression (HMM)")
            condition = st.selectbox("Condition to Model", list(report_predictions.keys()))
            observation_text = st.text_input(
                "Ordered Longitudinal Observations (earliest to latest, comma-separated)",
                value="NOT_MENTIONED, PRESENT, PRESENT"
            )
            prior_p = st.slider("Initial Prior P(Present)", 0.01, 0.99, 0.20, 0.01)

            c_t1, c_t2 = st.columns(2)
            absent_pers = c_t1.slider("P(Absent next | Absent now)", 0.50, 0.99, 0.95, 0.01)
            present_pers = c_t2.slider("P(Present next | Present now)", 0.50, 0.99, 0.85, 0.01)

            obs = [v.strip().upper() for v in observation_text.split(",") if v.strip()]
            if obs and all(o in REPORT_STATES for o in obs):
                trans_mat = np.array([[absent_pers, 1 - absent_pers], [1 - present_pers, present_pers]])
                emiss_mat = np.array([[0.03, 0.87, 0.10], [0.82, 0.08, 0.10]])
                obs_idx = [REPORT_STATES.index(o) for o in obs]

                posteriors, log_lik = hmm_forward_backward(
                    obs_idx, np.array([1 - prior_p, prior_p]), trans_mat, emiss_mat
                )
                hmm_df = pd.DataFrame(posteriors, columns=["P(condition absent)", "P(condition present)"])
                hmm_df.insert(0, "Study order", range(1, len(obs) + 1))
                hmm_df.insert(1, "Observation", obs)

                hmm_chart = create_hmm_trajectory_chart(hmm_df, condition)
                st.altair_chart(hmm_chart, width='stretch')
                st.dataframe(hmm_df, hide_index=True)

                with st.expander("📐 HMM Formula & Current Values", expanded=False):
                    st.markdown(
                        r"""
**Forward probability**
$$
\alpha_t(j)=e_j(o_t)\sum_i\alpha_{{t-1}}(i)a_{{ij}}
$$

**Posterior**
$$
\gamma_t(j)=\frac{{\alpha_t(j)\beta_t(j)}}
{{\sum_s\alpha_t(s)\beta_t(s)}}
$$

- Initial present prior: **{prior:.1%}**
- Absent → absent: **{absent_p:.1%}**
- Present → present: **{present_p:.1%}**
- Sequence likelihood: **$\log L={log_lik:.4f}$**
- Latest present posterior: **{latest:.1%}**
""".format(
                            prior=prior_p,
                            absent_p=absent_pers,
                            present_p=present_pers,
                            log_lik=log_lik,
                            latest=float(posteriors[-1, 1]),
                        )
                    )
            else:
                st.warning("Observations must contain only: PRESENT, ABSENT, or NOT_MENTIONED.")

        elif selected_method == "Sum-product algorithm":
            st.markdown("##### Binary Factor Graph Marginalization")
            sp_cond = st.selectbox("Condition Node", list(image_predictions.keys()))
            sp_prior = st.slider("Prior P(Condition Present)", 0.01, 0.99, 0.20, 0.01)
            sp_rel = st.slider("Report Evidence Reliability", 0.50, 0.99, 0.90, 0.01)

            img_s = float(image_predictions[sp_cond])
            rep_s = report_predictions.get(sp_cond, "NOT_MENTIONED")
            img_like = np.array([1 - img_s, img_s])

            if rep_s == "PRESENT":
                rep_like = np.array([1 - sp_rel, sp_rel])
            elif rep_s == "ABSENT":
                rep_like = np.array([sp_rel, 1 - sp_rel])
            else:
                rep_like = np.array([1.0, 1.0])

            posterior = binary_sum_product(sp_prior, [img_like, rep_like])

            fg_chart = create_factor_graph_flow_chart(sp_prior, img_s, rep_s, posterior)
            st.altair_chart(fg_chart, width='stretch')

            k1, k2 = st.columns(2)
            k1.metric("Updated Marginal: Present", f"{posterior[1]:.1%}")
            k2.metric("Updated Marginal: Absent", f"{posterior[0]:.1%}")

            with st.expander("📐 Sum-Product Formula & Current Values", expanded=False):
                st.markdown(
                    r"""
**Binary factor update**
$$
P(C=1\mid E)=
\frac{{P(C=1)\prod_f L_f(C=1)}}
{{\sum_{{c\in{{0,1}}}}P(C=c)\prod_f L_f(C=c)}}
$$

- Prior $P(C=1)$: **{prior:.1%}**
- Image likelihood: $[1-p_c,\;p_c]=[{absent_img:.1%},\;{present_img:.1%}]$
- Report evidence: **{report_status}**
- Updated present marginal: **{posterior_present:.1%}**
- Updated absent marginal: **{posterior_absent:.1%}**
""".format(
                        prior=sp_prior,
                        absent_img=float(img_like[0]),
                        present_img=float(img_like[1]),
                        report_status=rep_s,
                        posterior_present=float(posterior[1]),
                        posterior_absent=float(posterior[0]),
                    )
                )

        elif selected_method == "Supervised ML":
            st.markdown("##### Supervised Report Classification (CART, Random Forest, AdaBoost)")
            conditions = list(image_predictions.keys())
            target_cond = st.selectbox("Target Condition to Predict", conditions)
            ml_algo = st.selectbox("Ensemble Algorithm", ["Random forest", "AdaBoost", "CART"])

            c_ml1, c_ml2 = st.columns(2)
            test_frac = c_ml1.slider("Holdout Test Fraction", 0.1, 0.5, 0.2, 0.05)
            tree_depth = c_ml2.slider("Maximum Depth", 1, 6, 3)
            n_est = 25
            if ml_algo != "CART":
                n_est = st.slider("Number of Estimators", 5, 100, 25, 5)

            try:
                cohort = load_report_labels(labels_path)
                feats = [c for c in conditions if c != target_cond]
                r_ids, f_frame, targets = build_supervised_report_dataset(cohort, target_cond, feats)

                ml_res = run_cohort_classifier(
                    f_frame.to_numpy(),
                    targets,
                    tuple(f_frame.columns),
                    ml_algo,
                    test_frac,
                    tree_depth,
                    n_est,
                    CLASSIFIER_RESULT_VERSION,
                )

                m_cols = st.columns(5)
                for c_col, (m_name, m_val) in zip(m_cols, ml_res["metrics"].items()):
                    c_col.metric(m_name, f"{m_val:.1%}")

                # Confusion Matrix Heatmap
                c_mat = ml_res["confusion_matrix"]
                col_cm1, col_cm2 = st.columns([0.45, 0.55])
                with col_cm1:
                    cm_chart = create_confusion_matrix_heatmap(c_mat)
                    st.altair_chart(cm_chart, width='stretch')

                with col_cm2:
                    imp_df = pd.DataFrame({
                        "Feature": f_frame.columns,
                        "Importance": ml_res["feature_importances"]
                    }).sort_values("Importance", ascending=False).head(8)
                    st.markdown("**Top Predictors:**")
                    st.bar_chart(imp_df.set_index("Feature"))

                if ml_res.get("tree_rules"):
                    with st.expander("Decision Rules / Tree Logic"):
                        st.code(ml_res["tree_rules"])

                with st.expander("📐 Supervised ML Metrics", expanded=False):
                    accuracy = float(ml_res["metrics"].get("Accuracy", 0.0))
                    precision = float(ml_res["metrics"].get("Precision", 0.0))
                    recall = float(ml_res["metrics"].get("Recall", 0.0))
                    f1 = float(ml_res["metrics"].get("F1", 0.0))
                    st.markdown(
                        r"""
$$
\text{{Accuracy}}=\frac{{TP+TN}}{{TP+TN+FP+FN}}
$$
$$
\text{{Precision}}=\frac{{TP}}{{TP+FP}},\quad
\text{{Recall}}=\frac{{TP}}{{TP+FN}},\quad
F1=2\frac{{\text{{Precision}}\cdot\text{{Recall}}}}
{{\text{{Precision}}+\text{{Recall}}}}
$$

- Algorithm: **{algorithm}**
- Target: **{target}**
- Holdout fraction: **{fraction:.0%}**
- Accuracy: **{accuracy:.1%}**
- Precision: **{precision:.1%}**
- Recall: **{recall:.1%}**
- F1: **{f1:.1%}**
""".format(
                            algorithm=ml_algo,
                            target=target_cond,
                            fraction=test_frac,
                            accuracy=accuracy,
                            precision=precision,
                            recall=recall,
                            f1=f1,
                        )
                    )

            except Exception as e:
                st.error(f"Supervised model training error: {e}")

    # --------------------------------------------------------------------------
    # TAB 7: HOSPITAL COHORT POPULATION ANALYTICS (NEW!)
    # --------------------------------------------------------------------------
    with tab_cohort:
        st.markdown("#### 📊 Hospital Cohort Epidemiology & Dataset Explorer")
        st.caption(
            "Population-level analytics across the 7,471 Indiana University Chest X-Ray clinical studies."
        )

        try:
            cohort_df = load_report_labels(labels_path)
            c_conds = list(image_predictions.keys())

            col_pop1, col_pop2 = st.columns([0.5, 0.5])
            with col_pop1:
                prev_chart = create_cohort_prevalence_chart(cohort_df, c_conds)
                st.altair_chart(prev_chart, width='stretch')
            with col_pop2:
                co_chart = create_cooccurrence_heatmap(cohort_df, c_conds)
                st.altair_chart(co_chart, width='stretch')

            with st.expander("🔎 Sample Cohort Records (First 50 Reports)", expanded=False):
                st.dataframe(cohort_df.head(50), hide_index=True)

            with st.expander("📐 Cohort Prevalence Formula & Current Values", expanded=False):
                cohort_condition_rows = []
                for cohort_condition in c_conds:
                    if cohort_condition in cohort_df.columns:
                        present_count = int(
                            cohort_df[cohort_condition].astype(str).str.upper().eq("PRESENT").sum()
                        )
                        total_count = int(len(cohort_df))
                        prevalence = present_count / total_count if total_count else 0.0
                        cohort_condition_rows.append(
                            f"| {cohort_condition} | {present_count:,} | {total_count:,} | {prevalence:.1%} |"
                        )
                st.markdown(
                    r"""
$$
\text{{Prevalence}}_c=
\frac{{\text{{Number of reports where }}c=\text{{PRESENT}}}}
{{\text{{Total analyzed reports}}}}
$$

| Condition | PRESENT Count | Total Reports | Prevalence |
|---|---:|---:|---:|
{rows}

**Total records:** {total:,}
""".format(
                        rows="\n".join(cohort_condition_rows),
                        total=len(cohort_df),
                    )
                )

        except Exception as e:
            st.error(f"Cohort data loading error: {e}")


    # ==============================================================================
    # 13. AUDIT REPORT EXPORT
    # ==============================================================================
    st.markdown("---")
    st.markdown("### 📥 Adjudication Audit Report Export")

    audit_report_text = f"""# CLINICAL RADIOLOGY ADJUDICATION AUDIT REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Case Reference: {selected_scenario}
----------------------------------------------------------------------
CLAIM DETAILS:
- Claimed Condition: {claimed_cond}
- Claimed Severity:  {claimed_sev}
- Radiology Exam:    {rep_date}
- Claim Submission:  {clm_date}
- Temporal Lag:      {(clm_date - rep_date).days} days

ADJUDICATION RECOMMENDATION:
- Decision Verdict:  {decision}
- Risk Level:        {risk_level}
- Adjudicator Action:{explanation.get('recommended_action', 'N/A')}

FINDINGS CORRELATION:
{chr(10).join([f"- {r['condition']}: Image={r['image_score']:.1%} ({r['image_status']}), Report={r['report_status']} -> {r['correlation_result']}" for r in correlation_results])}

DECISION RATIONALE:
{chr(10).join([f"- {r}" for r in reasons]) if reasons else "- Evidence corroborates claim."}

SIGNATURE / AUDIT REVIEW:
Reviewing Adjudicator: ___________________________   Date: ____________
"""

    col_exp1, col_exp2 = st.columns([0.3, 0.7])
    with col_exp1:
        st.download_button(
            label="📄 Download Adjudication Audit Report (.md)",
            data=audit_report_text,
            file_name=f"adjudication_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            width='stretch'
        )
    with col_exp2:
        st.caption("Official immutable audit record including full evidence trace and reviewer signature fields.")


# ==============================================================================
# 14. INITIAL / EMPTY STATE
# ==============================================================================

else:
    st.info(
        "👋 **Welcome to the Radiology Correlation & Adjudication System.**\n\n"
        "1. Select a **Clinical Scenario Preset** from the sidebar (e.g. Confirmed Pneumonia, Discrepancy, Timeline Anomaly), "
        "or choose **Custom Upload**.\n"
        "2. Review the image and report text above.\n"
        "3. Click **Run Multi-Modal Correlation & Adjudicate Case** to generate complete analytical graphs and adjudication recommendations."
    )


# ==============================================================================
# 15. FOOTER
# ==============================================================================

st.divider()
st.caption(
    "Radiology Correlation System v2.5 · Decision-support tool for healthcare claims review · "
    "TorchXRayVision DenseNet-121 · Clinical NLP · Statutory Treatment Guidelines (STG) Engine"
)
