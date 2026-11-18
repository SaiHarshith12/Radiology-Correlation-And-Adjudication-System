import pandas as pd

from condition_extractor import extract_conditions


# Load paired dataset
df = pd.read_csv(
    "data/processed/iu_xray_pairs.csv"
)


# Test first 5 reports
for i in range(5):

    print("\n")
    print("=" * 60)

    print("REPORT ID:", df.loc[i, "report_id"])

    # Combine findings + impression
    findings = str(df.loc[i, "findings"])
    impression = str(df.loc[i, "impression"])

    report = findings + " " + impression

    print("\nREPORT:\n")
    print(report)

    print("\nEXTRACTED CONDITIONS:\n")

    results = extract_conditions(report)

    for condition, status in results.items():

        print(condition, ":", status)