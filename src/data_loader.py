import os
import xml.etree.ElementTree as ET
import pandas as pd

# Paths
REPORTS_FOLDER = "data/reports"
IMAGES_FOLDER = "data/images"

data = []

# Go through every XML report
for file_name in os.listdir(REPORTS_FOLDER):

    if not file_name.endswith(".xml"):
        continue

    file_path = os.path.join(REPORTS_FOLDER, file_name)

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        # -------------------------
        # REPORT ID
        # -------------------------
        uid = root.find("uId")

        if uid is None:
            continue

        report_id = uid.attrib.get("id")

        # -------------------------
        # FINDINGS + IMPRESSION
        # -------------------------
        findings = ""
        impression = ""

        for abstract in root.findall(".//AbstractText"):

            label = abstract.attrib.get("Label", "").upper()
            text = abstract.text or ""

            if label == "FINDINGS":
                findings = text.strip()

            elif label == "IMPRESSION":
                impression = text.strip()

        # -------------------------
        # IMAGE IDs
        # -------------------------
        images = root.findall("parentImage")

        for image in images:

            image_id = image.attrib.get("id")

            if not image_id:
                continue

            # IU X-Ray images are PNG files
            image_name = image_id + ".png"

            # Full image path
            image_path = os.path.join(
                IMAGES_FOLDER,
                image_name
            )

            # Add one row per image
            data.append({
                "report_id": report_id,
                "image_name": image_name,
                "image_path": image_path,
                "findings": findings,
                "impression": impression
            })

    except Exception as e:

        print(f"Error reading {file_name}: {e}")


# -------------------------
# CREATE DATAFRAME
# -------------------------

df = pd.DataFrame(data)

print("Total image-report pairs:", len(df))

print("\nFirst 5 rows:")
print(df.head())


# -------------------------
# SAVE CSV
# -------------------------

os.makedirs("data/processed", exist_ok=True)

df.to_csv(
    "data/processed/iu_xray_pairs.csv",
    index=False
)

print("\nDataset saved successfully!")