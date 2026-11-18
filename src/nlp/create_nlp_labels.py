import pandas as pd
import os

from condition_extractor import extract_conditions


# =========================================
# LOAD DATASET
# =========================================

INPUT_FILE = "data/processed/iu_xray_pairs.csv"

df = pd.read_csv(INPUT_FILE)

print("Total rows:", len(df))


# =========================================
# COMBINE FINDINGS + IMPRESSION
# =========================================

df["findings"] = df["findings"].fillna("")
df["impression"] = df["impression"].fillna("")

df["report_text"] = (
    df["findings"] + " " + df["impression"]
)


# =========================================
# APPLY NLP
# =========================================

print("\nApplying NLP to all reports...")

results = df["report_text"].apply(extract_conditions)


# Convert dictionary results into columns
labels_df = pd.DataFrame(results.tolist())


# =========================================
# COMBINE ORIGINAL DATA + NLP LABELS
# =========================================

final_df = pd.concat(
    [df, labels_df],
    axis=1
)


# =========================================
# SAVE RESULT
# =========================================

OUTPUT_FOLDER = "data/processed"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

OUTPUT_FILE = (
    "data/processed/"
    "iu_xray_nlp_labels.csv"
)

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nNLP processing completed!")

print("\nFirst 5 rows:")

print(
    final_df[
        [
            "report_id",
            "image_name",
            "Pneumonia",
            "Edema",
            "Pleural Effusion",
            "Pneumothorax",
            "Atelectasis",
            "Cardiomegaly"
        ]
    ].head()
)

print("\nFile saved at:")
print(OUTPUT_FILE)