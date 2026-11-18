# =========================================
# RADIOLOGY CORRELATION ENGINE
# =========================================


IMAGE_THRESHOLD = 0.50


def get_image_status(score):
    """
    Convert Image AI probability into
    a simple status.
    """

    if score >= IMAGE_THRESHOLD:
        return "PRESENT"

    return "NOT_DETECTED"


def correlate_condition(
    condition,
    image_score,
    report_status
):
    """
    Compare one Image AI prediction
    with one NLP report finding.
    """

    image_status = get_image_status(image_score)


    # -------------------------------------
    # CASE 1: REPORT SAYS PRESENT
    # -------------------------------------

    if report_status == "PRESENT":

        if image_status == "PRESENT":

            result = "CONSISTENT"

        else:

            result = "NEEDS_REVIEW"


    # -------------------------------------
    # CASE 2: REPORT SAYS ABSENT
    # -------------------------------------

    elif report_status == "ABSENT":

        if image_status == "PRESENT":

            result = "POTENTIAL_INCONSISTENCY"

        else:

            result = "CONSISTENT"


    # -------------------------------------
    # CASE 3: CONDITION NOT MENTIONED
    # -------------------------------------

    elif report_status == "NOT_MENTIONED":

        if image_status == "PRESENT":

            result = "NOT_MENTIONED_IN_REPORT"

        else:

            result = "NO_EVIDENCE"


    # -------------------------------------
    # UNKNOWN STATUS
    # -------------------------------------

    else:

        result = "UNKNOWN"


    return {

        "condition": condition,

        "image_score": round(
            float(image_score),
            3
        ),

        "image_status": image_status,

        "report_status": report_status,

        "correlation_result": result
    }


# =========================================
# CORRELATE ALL CONDITIONS
# =========================================


def correlate_all(
    image_predictions,
    report_predictions
):

    results = []


    for condition, image_score in image_predictions.items():

        report_status = report_predictions.get(
            condition,
            "NOT_MENTIONED"
        )


        result = correlate_condition(

            condition,

            image_score,

            report_status
        )


        results.append(result)


    return results