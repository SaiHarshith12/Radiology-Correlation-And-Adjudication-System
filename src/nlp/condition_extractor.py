import re


# =========================================
# CONDITIONS AND THEIR KEYWORDS
# =========================================

CONDITIONS = {

    "Pneumonia": [
        "pneumonia",
        "pneumonic",
        "focal consolidation",
        "focal airspace disease",
        "airspace disease"
    ],

    "Edema": [
        "pulmonary edema",
        "edema"
    ],

    "Pleural Effusion": [
        "pleural effusion",
        "pleural fluid"
    ],

    "Pneumothorax": [
        "pneumothorax"
    ],

    "Atelectasis": [
        "atelectasis",
        "atelectatic"
    ],

    "Cardiomegaly": [
        "cardiomegaly",
        "enlarged heart",
        "heart is enlarged",
        "cardiac enlargement"
    ]
}


# =========================================
# NEGATION PHRASES
# =========================================

NEGATIONS = [
    "no evidence of",
    "without evidence of",
    "negative for",
    "absence of",
    "free of",
    "without",
    "no"
]


# =========================================
# CHECK NEGATION
# =========================================

def is_negated(text, keyword_position):

    # Get up to 150 characters before condition
    before = text[max(0, keyword_position - 150):keyword_position]

    for negation in NEGATIONS:

        if negation in before:
            return True

    return False


# =========================================
# EXTRACT CONDITIONS
# =========================================

def extract_conditions(report_text):

    results = {}

    # Initialize all conditions
    for condition in CONDITIONS:
        results[condition] = "NOT_MENTIONED"

    if not isinstance(report_text, str):
        return results

    text = report_text.lower()

    # Check every condition
    for condition, keywords in CONDITIONS.items():

        for keyword in keywords:

            position = text.find(keyword)

            if position != -1:

                # Check whether this mention is negated
                if is_negated(text, position):

                    # Do not overwrite PRESENT
                    if results[condition] != "PRESENT":
                        results[condition] = "ABSENT"

                else:

                    results[condition] = "PRESENT"

    return results


# =========================================
# TEST
# =========================================

if __name__ == "__main__":

    sample_report = """
    The lungs are normally inflated without evidence
    of focal airspace disease, pleural effusion,
    or pneumothorax.
    """

    result = extract_conditions(sample_report)

    print("\nEXTRACTED CONDITIONS\n")

    for condition, status in result.items():

        print(condition, ":", status)