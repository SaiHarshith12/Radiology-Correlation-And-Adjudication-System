# =========================================
# FINAL ADJUDICATION ENGINE
# =========================================


def generate_adjudication_decision(

    claim_result,
    severity_result,
    timeline_result,
    stg_result

):

    # =========================================
    # GET RESULTS
    # =========================================

    claim_status = claim_result.get(
        "claim_status",
        "UNKNOWN"
    )

    severity_status = severity_result.get(
        "severity_status",
        "UNKNOWN"
    )

    timeline_status = timeline_result.get(
        "timeline_status",
        "UNKNOWN"
    )

    evidence_status = stg_result.get(
        "stg_status",
        "UNKNOWN"
    )


    # =========================================
    # DEFAULT RESULT
    # =========================================

    result = {

        "decision": "FLAG_FOR_HUMAN_REVIEW",

        "risk_level": "MEDIUM",

        "reasons": []

    }


    # =========================================
    # RULE 1
    # EVIDENCE INCONSISTENCY
    # =========================================

    if evidence_status == "REVIEW_REQUIRED":

        result["decision"] = (
            "FLAG_FOR_HUMAN_REVIEW"
        )

        result["risk_level"] = "HIGH"

        result["reasons"].append(

            "Evidence requires additional review"

        )


    # =========================================
    # RULE 2
    # CLAIM STATUS
    # =========================================

    if claim_status in [

        "POTENTIAL_INCONSISTENCY",
        "FLAG_FOR_HUMAN_REVIEW"

    ]:

        result["decision"] = (
            "FLAG_FOR_HUMAN_REVIEW"
        )

        result["risk_level"] = "HIGH"

        result["reasons"].append(

            "Claim evidence is inconsistent"

        )


    # =========================================
    # RULE 3
    # TIMELINE
    # =========================================

    if timeline_status not in [

        "CONSISTENT",
        "INSUFFICIENT_DATA"

    ]:

        result["decision"] = (
            "FLAG_FOR_HUMAN_REVIEW"
        )

        result["reasons"].append(

            "Timeline requires review"

        )


    # =========================================
    # RULE 4
    # ALL EVIDENCE CONSISTENT
    # =========================================

    if (

        evidence_status == "EVIDENCE_CONSISTENT"

        and

        timeline_status == "CONSISTENT"

    ):

        result["decision"] = (
            "EVIDENCE_CONSISTENT"
        )

        result["risk_level"] = "LOW"

        result["reasons"].append(

            "Image, report and timeline evidence are consistent"

        )


    # =========================================
    # RETURN RESULT
    # =========================================

    return result