import pandas as pd
import os

# Load dataset
df = pd.read_csv("data/processed/iu_xray_pairs.csv")

print("Total rows:", len(df))

# Check first 10 image paths
print("\nChecking first 10 image paths:\n")

for i in range(10):

    image_path = df.loc[i, "image_path"]

    exists = os.path.exists(image_path)

    print(
        df.loc[i, "image_name"],
        "->",
        exists
    )


# Count valid images
df["image_exists"] = df["image_path"].apply(os.path.exists)

print("\nValid images:", df["image_exists"].sum())

print("Missing images:", (~df["image_exists"]).sum())