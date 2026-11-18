# =========================================
# STG / EVIDENCE RULE ENGINE
# =========================================


def check_evidence_rules(
    claimed_condition,
    claim_result,
    timeline_result
):

    # =========================================
    # GET CLAIM INFORMATION
    # =========================================

    claim_status = claim_result.get(
        "claim_status",
        "UNKNOWN"
    )

    image_score = claim_result.get(
        "image_score",
        0
    )

    correlation = claim_result.get(
        "correlation_result",
        "UNKNOWN"
    )

    timeline_status = timeline_result.get(
        "timeline_status",
        "UNKNOWN"
    )


    # =========================================
    # DEFAULT RESULT
    # =========================================

    result = {

        "condition": claimed_condition,

        "stg_status": "REVIEW_REQUIRED",

        "rule_results": []

    }


    # =========================================
    # RULE 1
    # IMAGE EVIDENCE
    # =========================================

    if image_score >= 0.50:

        result["rule_results"].append(

            "Image evidence available"

        )

    else:

        result["rule_results"].append(

            "Limited image evidence"

        )


    # =========================================
    # RULE 2
    # REPORT CORRELATION
    # =========================================

    if correlation == "CONSISTENT":

        result["rule_results"].append(

            "Image and report are consistent"

        )

    elif correlation == "POTENTIAL_INCONSISTENCY":

        result["rule_results"].append(

            "Image and report require review"

        )

        result["stg_status"] = (
            "REVIEW_REQUIRED"
        )


    else:

        result["rule_results"].append(

            "No sufficient correlation evidence"

        )


    # =========================================
    # RULE 3
    # TIMELINE
    # =========================================

    if timeline_status == "CONSISTENT":

        result["rule_results"].append(

            "Timeline is consistent"

        )

    elif timeline_status == "INSUFFICIENT_DATA":

        result["rule_results"].append(

            "Timeline data unavailable"

        )

    else:

        result["rule_results"].append(

            "Timeline requires review"

        )

        result["stg_status"] = (
            "REVIEW_REQUIRED"
        )


    # =========================================
    # FINAL EVIDENCE STATUS
    # =========================================

    if (

        image_score >= 0.50

        and

        correlation == "CONSISTENT"

        and

        timeline_status == "CONSISTENT"

    ):

        result["stg_status"] = (
            "EVIDENCE_CONSISTENT"
        )


    return result