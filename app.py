
import json
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Satellite AI | Land Cover",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Satellite AI Land-Cover Intelligence")
st.caption(
    "Deep learning prototype | EuroSAT dataset | ResNet18"
)

st.markdown("---")

# ---------------- MODEL SETTINGS ----------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "satellite_resnet18.pth"
CLASS_PATH = ROOT / "models" / "class_names.json"

DEVICE = torch.device("cpu")


@st.cache_resource
def load_model():
    with open(CLASS_PATH, "r") as f:
        class_names = json.load(f)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(
        model.fc.in_features,
        len(class_names)
    )

    state = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(state)
    model.eval()

    return model, class_names


# ---------------- IMAGE TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- MAIN APP ----------------
left, right = st.columns([1, 1])

with left:
    st.subheader("📤 Upload Satellite Image")

    uploaded_file = st.file_uploader(
        "Choose a JPG or PNG image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(
            image,
            caption="Uploaded satellite image",
            use_container_width=True
        )

with right:
    st.subheader("🔍 AI Classification")

    if uploaded_file:
        if st.button(
            "🚀 Analyze Image",
            type="primary",
            use_container_width=True
        ):
            try:
                with st.spinner("Analyzing satellite image..."):
                    model, class_names = load_model()

                    tensor = transform(image).unsqueeze(0)

                    with torch.no_grad():
                        outputs = model(tensor)
                        probabilities = torch.softmax(
                            outputs, dim=1
                        )[0]

                    top_values, top_indices = torch.topk(
                        probabilities,
                        k=min(3, len(class_names))
                    )

                    predicted_class = class_names[
                        top_indices[0].item()
                    ]
                    confidence = top_values[0].item() * 100

                st.success(f"Predicted Land Cover: {predicted_class}")

                st.metric(
                    "Model confidence",
                    f"{confidence:.2f}%"
                )

                st.markdown("### Top predictions")

                for value, index in zip(
                    top_values.tolist(),
                    top_indices.tolist()
                ):
                    st.write(
                        f"**{class_names[index]}** — "
                        f"{value * 100:.2f}%"
                    )
                    st.progress(float(value))

                st.info(
                    "This is a research prototype. "
                    "Confidence is not a guarantee of correctness."
                )

            except FileNotFoundError:
                st.error(
                    "Model files were not found. "
                    "Check the models folder and filenames."
                )

            except Exception as e:
                st.error(f"Could not analyze image: {e}")

    else:
        st.info("Upload an image to begin analysis.")

st.markdown("---")
st.caption(
    "Dataset: EuroSAT | Model: ResNet18 | "
    "Prototype for land-cover classification"
)
