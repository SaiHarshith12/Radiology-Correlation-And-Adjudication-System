# =========================================
# SEVERITY VERIFICATION MODULE
# =========================================


def estimate_severity(image_score):

    """
    Prototype severity estimation.

    IMPORTANT:
    Model confidence is NOT the same as clinical severity.
    This is only a demo-level evidence-strength estimate.
    """

    if image_score >= 0.80:
        return "STRONG_EVIDENCE"

    elif image_score >= 0.50:
        return "MODERATE_EVIDENCE"

    elif image_score >= 0.30:
        return "LOW_EVIDENCE"

    else:
        return "NO_STRONG_EVIDENCE"


def verify_severity(claimed_severity, image_score):

    evidence_level = estimate_severity(
        image_score
    )


    # =====================================
    # PROTOTYPE COMPARISON
    # =====================================

    if claimed_severity == "MILD":

        if evidence_level in [
            "LOW_EVIDENCE",
            "MODERATE_EVIDENCE",
            "STRONG_EVIDENCE"
        ]:

            result = "REVIEW"

        else:

            result = "INSUFFICIENT_EVIDENCE"


    elif claimed_severity == "MODERATE":

        if evidence_level in [
            "MODERATE_EVIDENCE",
            "STRONG_EVIDENCE"
        ]:

            result = "REVIEW"

        else:

            result = "POTENTIAL_INCONSISTENCY"


    elif claimed_severity == "SEVERE":

        if evidence_level == "STRONG_EVIDENCE":

            result = "REVIEW"

        else:

            result = "POTENTIAL_INCONSISTENCY"


    else:

        result = "UNKNOWN_SEVERITY"


    return {

        "claimed_severity": claimed_severity,

        "image_score": image_score,

        "evidence_level": evidence_level,

        "severity_result": result

    }