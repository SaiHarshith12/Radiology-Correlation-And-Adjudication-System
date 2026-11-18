import pandas as pd


# Load NLP-labelled dataset
df = pd.read_csv(
    "data/processed/iu_xray_nlp_labels.csv"
)


# Conditions
conditions = [
    "Pneumonia",
    "Edema",
    "Pleural Effusion",
    "Pneumothorax",
    "Atelectasis",
    "Cardiomegaly"
]


print("\nTOTAL DATASET ROWS:", len(df))

print("\n" + "=" * 50)
print("CONDITION DISTRIBUTION")
print("=" * 50)


for condition in conditions:

    print(f"\n{condition}")

    counts = df[condition].value_counts()

    print(counts)

    present_count = (
        df[condition] == "PRESENT"
    ).sum()

    absent_count = (
        df[condition] == "ABSENT"
    ).sum()

    not_mentioned_count = (
        df[condition] == "NOT_MENTIONED"
    ).sum()

    print("\nSummary:")

    print("PRESENT:", present_count)
    print("ABSENT:", absent_count)
    print("NOT MENTIONED:", not_mentioned_count)


print("\n" + "=" * 50)
print("ANALYSIS COMPLETED")
print("=" * 50)