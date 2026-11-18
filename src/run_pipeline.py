import sys
sys.stdout.reconfigure(encoding="utf-8")


# =========================================
# IMPORT PROJECT MODULES
# =========================================

from image_model.image_predictor import predict_image

from nlp.condition_extractor import extract_conditions

from correlation.correlation_engine import correlate_all

from claim.claim_verifier import verify_claim

from explanation.adjudicator_explainer import generate_explanation

from severity.severity_verifier import verify_severity

from input.claim_input import get_claim_input

from timeline.timeline_verifier import verify_timeline

from stg.stg_engine import check_evidence_rules

from adjudication.adjudication_engine import (
    generate_adjudication_decision
)


# =========================================
# INPUT DATA
# =========================================

IMAGE_PATH = (
    "data/images/CXR1_1_IM-0001-3001.png"
)


REPORT_TEXT = """

The cardiac silhouette and mediastinum size
are within normal limits.

There is no pulmonary edema.

There is no focal consolidation.

There are no signs of a pleural effusion.

There is no evidence of pneumothorax.

Normal chest x-ray.

"""

# =========================================
# GET CLAIM INFORMATION
# =========================================

claim_data = get_claim_input()


CLAIMED_CONDITION = (
    claim_data["claimed_condition"]
)


CLAIMED_SEVERITY = (
    claim_data["claimed_severity"]
)


CLAIM_DATE = (
    claim_data["claim_date"]
)


REPORT_DATE = (
    claim_data["report_date"]
)



# =========================================
# STEP 1: IMAGE ANALYSIS
# =========================================

print("\n" + "=" * 65)

print("STEP 1: ANALYZING CHEST X-RAY")

print("=" * 65)


image_predictions = predict_image(

    IMAGE_PATH

)
# =========================================
# DISPLAY IMAGE AI RESULTS
# =========================================

print("\n" + "=" * 65)

print("IMAGE AI ANALYSIS RESULTS")

print("=" * 65)


for condition, score in image_predictions.items():

    print(

        condition,

        ":",

        round(score, 3)

    )

# =========================================
# STEP 2: REPORT NLP
# =========================================

print("\n" + "=" * 65)

print("STEP 2: ANALYZING RADIOLOGY REPORT")

print("=" * 65)


report_predictions = extract_conditions(

    REPORT_TEXT

)


# Display report results

for condition, status in report_predictions.items():

    print(

        condition,

        ":",

        status

    )


# =========================================
# STEP 3: CORRELATION
# =========================================

print("\n" + "=" * 65)

print("STEP 3: CORRELATING IMAGE AND REPORT")

print("=" * 65)


correlation_results = correlate_all(

    image_predictions,

    report_predictions

)

# =========================================
# STEP 4: CLAIM VERIFICATION
# =========================================

print("\n" + "=" * 65)

print("STEP 4: VERIFYING CLAIMED CONDITION")

print("=" * 65)


claim_result = verify_claim(

    CLAIMED_CONDITION,

    correlation_results

)

# =========================================
# STEP 5: GENERATE ADJUDICATOR EXPLANATION
# =========================================

adjudicator_explanation = generate_explanation(

    claim_result

)

# =========================================
# STEP 6: SEVERITY EVIDENCE CHECK
# =========================================

claimed_image_score = claim_result.get(
    "image_score",
    0
)


severity_result = verify_severity(

    CLAIMED_SEVERITY,

    claimed_image_score

)
# =========================================
# STEP 7: TIMELINE VERIFICATION
# =========================================

print("\n" + "=" * 65)

print("STEP 7: TIMELINE VERIFICATION")

print("=" * 65)


timeline_result = verify_timeline(

    REPORT_DATE,

    CLAIM_DATE

)

# =========================================
# STEP 8: STG / EVIDENCE CHECK
# =========================================

stg_result = check_evidence_rules(

    CLAIMED_CONDITION,

    claim_result,

    timeline_result

)

# =========================================
# STEP 9: FINAL ADJUDICATION
# =========================================

print("\n" + "=" * 65)

print("STEP 9: FINAL ADJUDICATION")

print("=" * 65)


adjudication_result = generate_adjudication_decision(

    claim_result,

    severity_result,

    timeline_result,

    stg_result

)


# =========================================
# FINAL RESULTS
# =========================================

print("\n" + "=" * 65)

print("FINAL CORRELATION RESULTS")

print("=" * 65)


for result in correlation_results:

    print("\nCondition:", result["condition"])

    print(

        "Image Score:",

        result["image_score"]

    )

    print(

        "Image Status:",

        result["image_status"]

    )

    print(

        "Report Status:",

        result["report_status"]

    )

    print(

        "Correlation:",

        result["correlation_result"]

    )

    print("-" * 65)

# =========================================
# CLAIM VERIFICATION RESULT
# =========================================

print("\n" + "=" * 65)

print("CLAIM VERIFICATION RESULT")

print("=" * 65)


print(

    "\nClaimed Condition:",

    claim_result["claimed_condition"]

)


print(

    "Image Evidence Score:",

    claim_result.get(
        "image_score",
        "N/A"
    )

)


print(

    "Report Finding:",

    claim_result.get(
        "report_status",
        "N/A"
    )

)


print(

    "Correlation:",

    claim_result.get(
        "correlation_result",
        "N/A"
    )

)


print(

    "\nFINAL CLAIM STATUS:",

    claim_result["claim_status"]

)


print(

    "\nExplanation:",

    claim_result["message"]

)

# =========================================
# ADJUDICATOR EXPLANATION
# =========================================

print("\n" + "=" * 65)

print("ADJUDICATOR DECISION SUPPORT")

print("=" * 65)


print(

    "\nClaimed Condition:",

    adjudicator_explanation[
        "claimed_condition"
    ]

)


print(

    "Image Evidence Score:",

    round(
        adjudicator_explanation[
            "image_score"
        ],
        3
    )

)


print(

    "AI Confidence Level:",

    adjudicator_explanation[
        "confidence"
    ]

)


print(

    "Report Finding:",

    adjudicator_explanation[
        "report_status"
    ]

)


print(

    "System Decision:",

    adjudicator_explanation[
        "claim_status"
    ]

)


print(

    "\nReason:",

    adjudicator_explanation[
        "reason"
    ]

)


print(

    "\nRecommended Action:",

    adjudicator_explanation[
        "recommended_action"
    ]

)

# =========================================
# SEVERITY EVIDENCE RESULT
# =========================================

print("\n" + "=" * 65)

print("SEVERITY / EVIDENCE CHECK")

print("=" * 65)


print(

    "\nClaimed Severity:",

    severity_result[
        "claimed_severity"
    ]

)


print(

    "Image Evidence Score:",

    round(
        severity_result[
            "image_score"
        ],
        3
    )

)


print(

    "Image Evidence Level:",

    severity_result[
        "evidence_level"
    ]

)


print(

    "\nSeverity Check Result:",

    severity_result[
        "severity_result"
    ]

)

# =========================================
# TIMELINE VERIFICATION RESULT
# =========================================

print("\n" + "=" * 65)

print("TIMELINE VERIFICATION RESULT")

print("=" * 65)


print(

    "\nReference/Report Date:",

    timeline_result.get(
        "report_date",
        "N/A"
    )

)


print(

    "Claim Date:",

    timeline_result.get(
        "claim_date",
        "N/A"
    )

)


print(

    "Difference:",

    timeline_result.get(
        "difference_days",
        "N/A"
    ),

    "days"

)


print(

    "\nTimeline Status:",

    timeline_result[
        "timeline_status"
    ]

)


print(

    "\nExplanation:",

    timeline_result[
        "message"
    ]

)

# =========================================
# STG / EVIDENCE RESULT
# =========================================

print("\n" + "=" * 65)

print("STG / EVIDENCE CONSISTENCY CHECK")

print("=" * 65)


print(

    "\nCondition:",

    stg_result["condition"]

)


print(

    "\nRule Results:"

)


for rule in stg_result["rule_results"]:

    print(

        "-",

        rule

    )


print(

    "\nFINAL EVIDENCE STATUS:",

    stg_result["stg_status"]

)

# =========================================
# FINAL ADJUDICATION RESULT
# =========================================

print("\n" + "=" * 65)

print("FINAL ADJUDICATION RESULT")

print("=" * 65)


print(

    "\nDECISION:",

    adjudication_result["decision"]

)


print(

    "\nRISK LEVEL:",

    adjudication_result["risk_level"]

)


print(

    "\nREASONS:"

)


for reason in adjudication_result["reasons"]:

    print(

        "-",

        reason

    )


print("\nRECOMMENDATION:")


if (

    adjudication_result["decision"]

    == "FLAG_FOR_HUMAN_REVIEW"

):

    print(

        "Potential inconsistency detected."
    )

    print(

        "Forward case to human adjudicator."
    )


else:

    print(

        "Evidence appears consistent."
    )

    print(

        "Human adjudicator can proceed with review."
    )