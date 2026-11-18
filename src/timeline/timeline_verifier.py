# =========================================
# TIMELINE VERIFICATION MODULE
# =========================================

from datetime import datetime


def verify_timeline(report_date, claim_date):

    # =========================================
    # CHECK MISSING DATES
    # =========================================

    if not report_date:

        return {

            "timeline_status": "INSUFFICIENT_DATA",

            "message": (
                "Report/reference date is not "
                "available for timeline verification."
            )

        }

    if not claim_date:

        return {

            "timeline_status": "INSUFFICIENT_DATA",

            "message": (
                "Claim date is required for "
                "timeline verification."
            )

        }


    # =========================================
    # CONVERT STRING TO DATE
    # =========================================

    try:

        report_datetime = datetime.strptime(
            report_date,
            "%Y-%m-%d"
        )

        claim_datetime = datetime.strptime(
            claim_date,
            "%Y-%m-%d"
        )

    except ValueError:

        return {

            "timeline_status": "INVALID_DATE",

            "message": (
                "Invalid date format. "
                "Please use YYYY-MM-DD."
            )

        }


    # =========================================
    # CALCULATE DATE DIFFERENCE
    # =========================================

    difference = (
        claim_datetime -
        report_datetime
    ).days


    # =========================================
    # TIMELINE RULES
    # =========================================

    if difference < 0:

        timeline_status = (
            "POTENTIAL_INCONSISTENCY"
        )

        message = (
            "The claim date occurs before "
            "the reference report date."
        )


    elif difference == 0:

        timeline_status = "CONSISTENT"

        message = (
            "The claim and reference report "
            "occurred on the same day."
        )


    elif difference <= 30:

        timeline_status = "CONSISTENT"

        message = (
            "The reference report occurred "
            "before the claim within the "
            "configured review window."
        )


    else:

        timeline_status = "NEEDS_REVIEW"

        message = (
            "There is a significant time gap "
            "between the reference report "
            "and the claim."
        )


    # =========================================
    # RETURN RESULT
    # =========================================

    return {

        "report_date": report_date,

        "claim_date": claim_date,

        "difference_days": difference,

        "timeline_status": timeline_status,

        "message": message

    }