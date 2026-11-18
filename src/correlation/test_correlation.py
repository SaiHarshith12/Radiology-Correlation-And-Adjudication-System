from correlation_engine import correlate_all


# =========================================
# SAMPLE IMAGE AI PREDICTIONS
# =========================================

image_predictions = {

    "Pneumonia": 0.746,

    "Edema": 0.348,

    "Pleural Effusion": 0.364,

    "Pneumothorax": 0.305,

    "Atelectasis": 0.263,

    "Cardiomegaly": 0.045
}


# =========================================
# SAMPLE REPORT NLP RESULTS
# =========================================

report_predictions = {

    "Pneumonia": "NOT_MENTIONED",

    "Edema": "ABSENT",

    "Pleural Effusion": "ABSENT",

    "Pneumothorax": "ABSENT",

    "Atelectasis": "NOT_MENTIONED",

    "Cardiomegaly": "NOT_MENTIONED"
}


# =========================================
# RUN CORRELATION
# =========================================

results = correlate_all(

    image_predictions,

    report_predictions
)


# =========================================
# DISPLAY RESULTS
# =========================================

print("\n" + "=" * 65)

print("RADIOLOGY IMAGE-REPORT CORRELATION RESULTS")

print("=" * 65)


for result in results:

    print()

    print("Condition:", result["condition"])

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