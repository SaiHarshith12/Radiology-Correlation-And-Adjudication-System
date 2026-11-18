# =========================================
# CLAIM VERIFICATION MODULE
# =========================================


def verify_claim(
    claimed_condition,
    correlation_results
):

    # -------------------------------------
    # FIND THE CLAIMED CONDITION
    # -------------------------------------

    claimed_result = None


    for result in correlation_results:

        if result["condition"].lower() == claimed_condition.lower():

            claimed_result = result

            break


    # -------------------------------------
    # CONDITION NOT FOUND
    # -------------------------------------

    if claimed_result is None:

        return {

            "claimed_condition": claimed_condition,

            "claim_status": "CONDITION_NOT_SUPPORTED",

            "message": (
                "The claimed condition is not "
                "currently supported by the system."
            )
        }


    # -------------------------------------
    # GET CORRELATION RESULT
    # -------------------------------------

    correlation = claimed_result[
        "correlation_result"
    ]

    image_score = claimed_result[
        "image_score"
    ]

    report_status = claimed_result[
        "report_status"
    ]


    # =====================================
    # DECISION LOGIC
    # =====================================


    # CASE 1
    # Image and report support condition

    if correlation == "CONSISTENT":

        # If report explicitly says PRESENT
        if report_status == "PRESENT":

            claim_status = "SUPPORTED"

            message = (
                "Both the image analysis and "
                "radiology report are consistent "
                "with the claimed condition."
            )

        else:

            claim_status = "NOT_SUPPORTED"

            message = (
                "The radiology report indicates "
                "that the claimed condition is absent."
            )


    # CASE 2
    # Image-report conflict

    elif correlation == "POTENTIAL_INCONSISTENCY":

        claim_status = "NEEDS_REVIEW"

        message = (
            "The image analysis and radiology report "
            "provide conflicting evidence. "
            "Human adjudicator review is required."
        )


    # CASE 3
    # Report does not mention it

    elif correlation == "NOT_MENTIONED_IN_REPORT":

        claim_status = "NEEDS_REVIEW"

        message = (
            "The image analysis suggests the condition, "
            "but it is not explicitly mentioned in "
            "the radiology report."
        )


    # CASE 4
    # No evidence from image

    elif correlation == "NO_EVIDENCE":

        claim_status = "INSUFFICIENT_EVIDENCE"

        message = (
            "The image analysis does not provide "
            "strong evidence for the claimed condition."
        )


    # CASE 5
    # Report says present but image does not strongly detect

    elif correlation == "NEEDS_REVIEW":

        claim_status = "NEEDS_REVIEW"

        message = (
            "The report mentions the claimed condition, "
            "but image analysis does not strongly "
            "support it."
        )


    # UNKNOWN

    else:

        claim_status = "UNKNOWN"

        message = (
            "Unable to determine claim consistency."
        )


    # =====================================
    # RETURN RESULT
    # =====================================

    return {

        "claimed_condition": claimed_condition,

        "image_score": image_score,

        "report_status": report_status,

        "correlation_result": correlation,

        "claim_status": claim_status,

        "message": message
    }