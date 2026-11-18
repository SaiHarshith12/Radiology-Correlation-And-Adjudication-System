# =========================================
# ADJUDICATOR EXPLANATION MODULE
# =========================================


def get_confidence_level(score):

    if score >= 0.70:
        return "HIGH"

    elif score >= 0.50:
        return "MEDIUM"

    elif score >= 0.30:
        return "LOW"

    else:
        return "VERY_LOW"


def generate_explanation(claim_result):

    claimed_condition = claim_result["claimed_condition"]

    image_score = claim_result.get(
        "image_score",
        0
    )

    report_status = claim_result.get(
        "report_status",
        "NOT_MENTIONED"
    )

    claim_status = claim_result.get(
        "claim_status",
        "UNKNOWN"
    )


    # =========================================
    # CONFIDENCE
    # =========================================

    confidence = get_confidence_level(
        image_score
    )


    # =========================================
    # DECISION EXPLANATION
    # =========================================

    if claim_status == "SUPPORTED":

        reason = (
            f"The image analysis provides {confidence.lower()} "
            f"evidence for {claimed_condition}, and the radiology "
            f"report also supports the condition."
        )

        action = (
            "The case appears consistent. "
            "Proceed with normal adjudicator review."
        )


    elif claim_status == "NEEDS_REVIEW":

        reason = (
            f"The image AI provides {confidence.lower()} evidence "
            f"for {claimed_condition}, while the radiology report "
            f"indicates: {report_status}."
        )

        action = (
            "Flag this case for human adjudicator review."
        )


    elif claim_status == "NOT_SUPPORTED":

        reason = (
            f"The available evidence does not support "
            f"the claimed condition: {claimed_condition}."
        )

        action = (
            "Review the claim details and supporting documents."
        )


    elif claim_status == "INSUFFICIENT_EVIDENCE":

        reason = (
            f"The image analysis provides insufficient evidence "
            f"for {claimed_condition}."
        )

        action = (
            "Additional clinical or radiological evidence "
            "may be required."
        )


    else:

        reason = (
            "The system could not determine a clear decision "
            "from the available evidence."
        )

        action = (
            "Manual adjudicator review is recommended."
        )


    # =========================================
    # RETURN EXPLANATION
    # =========================================

    return {

        "claimed_condition": claimed_condition,

        "image_score": image_score,

        "confidence": confidence,

        "report_status": report_status,

        "claim_status": claim_status,

        "reason": reason,

        "recommended_action": action

    }