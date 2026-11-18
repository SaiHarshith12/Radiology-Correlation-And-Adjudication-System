# =========================================
# CLAIM INPUT MODULE
# =========================================


# =========================================
# CLAIM INPUT MODULE
# =========================================


def get_claim_input():

    print("\n" + "=" * 65)

    print("CLAIM INFORMATION INPUT")

    print("=" * 65)


    # =========================================
    # CLAIMED CONDITION
    # =========================================

    claimed_condition = input(

        "\nEnter Claimed Condition: "

    ).strip().title()


    # =========================================
    # CLAIMED SEVERITY
    # =========================================

    claimed_severity = input(

        "Enter Claimed Severity "
        "(MILD / MODERATE / SEVERE): "

    ).strip().upper()


    # =========================================
    # CLAIM DATE
    # =========================================

    claim_date = input(

        "Enter Claim Date "
        "(YYYY-MM-DD): "

    ).strip()


    # =========================================
    # REPORT DATE
    # =========================================

    report_date = input(

        "Enter Report/Reference Date "
        "(YYYY-MM-DD): "

    ).strip()


    return {

        "claimed_condition": claimed_condition,

        "claimed_severity": claimed_severity,

        "claim_date": claim_date,

        "report_date": report_date

    }


    # =========================================
    # CLAIMED CONDITION
    # =========================================

    claimed_condition = input(
        "\nEnter Claimed Condition: "
    )


    # =========================================
    # CLAIMED SEVERITY
    # =========================================

    claimed_severity = input(
        "Enter Claimed Severity "
        "(MILD / MODERATE / SEVERE): "
    )


    # =========================================
    # CLAIM DATE
    # =========================================

    claim_date = input(
        "Enter Claim Date "
        "(YYYY-MM-DD): "
    )


    # =========================================
    # REFERENCE DATE
    # =========================================

    report_date = input(
        "Enter Report/Reference Date "
        "(YYYY-MM-DD): "
    )


    # =========================================
    # RETURN DATA
    # =========================================

    return {

        "claimed_condition": claimed_condition,

        "claimed_severity": claimed_severity,

        "claim_date": claim_date,

        "report_date": report_date

    }