import sys

sys.stdout.reconfigure(encoding="utf-8")
import torch
import torchxrayvision as xrv


# =========================================
# IMAGE PATH
# =========================================

IMAGE_PATH = (
    "data/images/CXR1_1_IM-0001-3001.png"
)


# =========================================
# LOAD IMAGE
# =========================================

print("Loading image...")

img = xrv.utils.load_image(IMAGE_PATH)


# =========================================
# PREPROCESS IMAGE
# =========================================

transform = xrv.datasets.XRayCenterCrop()

img = transform(img)

resize = xrv.datasets.XRayResizer(224)

img = resize(img)


# Convert to PyTorch tensor
img_tensor = torch.from_numpy(img).unsqueeze(0)


print("Image shape:", img_tensor.shape)


# =========================================
# LOAD PRETRAINED MODEL
# =========================================

print("\nLoading pretrained Chest X-ray model...")

model = xrv.models.DenseNet(
    weights="densenet121-res224-all"
)

model.eval()
# Use CPU for inference
device = torch.device("cpu")

model = model.to(device)
img_tensor = img_tensor.to(device)


# =========================================
# RUN PREDICTION
# =========================================

print("\nAnalyzing X-ray...")

with torch.no_grad():

    outputs = model(img_tensor)


# =========================================
# DISPLAY RESULTS
# =========================================

predictions = dict(
    zip(
        model.pathologies,
        outputs[0].cpu().numpy()
    )
)


print("\n" + "=" * 50)

print("CHEST X-RAY AI PREDICTIONS")

print("=" * 50)


for pathology, score in predictions.items():

    print(
        pathology,
        ":",
        round(float(score), 3)
    )
# =========================================
# PROJECT CONDITION MAPPING
# =========================================

PROJECT_MAPPING = {

    "Pneumonia": "Pneumonia",
    "Edema": "Edema",
    "Pleural Effusion": "Effusion",
    "Pneumothorax": "Pneumothorax",
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly"
}


# Create project predictions
project_predictions = {}


for project_condition, model_condition in PROJECT_MAPPING.items():

    project_predictions[project_condition] = float(
        predictions.get(model_condition, 0)
    )


print("\n" + "=" * 50)

print("PROJECT CONDITION PREDICTIONS")

print("=" * 50)


for condition, score in project_predictions.items():

    print(
        condition,
        ":",
        round(score, 3)
    )