import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from pathlib import Path
import torch
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self, out_1: int = 16, out_2: int = 32):
        super().__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(out_2 * 7 * 7, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)

        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

def load_model_state(model: torch.nn.Module, path: str | Path, map_location: torch.device) -> torch.nn.Module:
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
    return model

def _to_28x28_grayscale(img: Image.Image) -> Image.Image:
    """
    Ensures a 28x28 grayscale image (MNIST-like).
    """
    img = img.convert("L")               # grayscale
    img = img.resize((28, 28))           # simple resize (ok for demo)
    return img

def preprocess_pil(pil_img: Image.Image) -> torch.Tensor:
    pil_img = pil_img.convert("L").resize((28, 28))
    arr = np.array(pil_img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


@st.cache_resource
def load_trained_model(weights_path: str, out_1: int = 16, out_2: int = 32):
    """Carga el modelo y los pesos. Cacheado para que no cargue en cada interacción."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(out_1=out_1, out_2=out_2).to(device)

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device

def predict(model: nn.Module, x: torch.Tensor, device: torch.device):
    """Devuelve predicción (int) y probabilidades (np.array de 10)."""
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)  # [1,10]
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred = int(np.argmax(probs))
    return pred, probs

# -----------------------------
# 3) Streamlit UI
# -----------------------------
st.set_page_config(page_title="MNIST Classifier (Techy)", layout="wide")
st.title("MNIST Classifier — CNN + Demo")
st.caption("Dibuja un dígito o sube una imagen. El modelo predice 0–9 y muestra probabilidades.")

# Ruta del modelo (mismo directorio que app.py)
WEIGHTS = Path(__file__).resolve().parent / "mnist_cnn.pt"

if not WEIGHTS.exists():
    st.error("No encuentro `mnist_cnn.pt` en el repo. Súbelo al mismo nivel que `app.py`.")
    st.stop()

left, right = st.columns([1, 1])

with left:
    st.subheader("1) Entrada")

    mode = st.radio("Elige modo:", ["🖊️ Dibujar", "🖼️ Subir imagen"], horizontal=True)

    pil_img = None

    if mode == "🖊️ Dibujar":
        st.write("Dibuja un dígito (0–9). Ideal: trazo grueso y centrado.")

        if st_canvas is None:
            st.warning("No está instalado `streamlit-drawable-canvas`. Usa 'Subir imagen' o agrega la dependencia.")
        else:
            canvas = st_canvas(
                fill_color="black",
                stroke_width=18,
                stroke_color="white",
                background_color="black",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
            )

            if canvas.image_data is not None:
                st.write("canvas dtype:", canvas.image_data.dtype,
                         "min:", float(np.min(canvas.image_data)),
                         "max:", float(np.max(canvas.image_data)))
            
            # 1) Detectar si hay trazos (lo más confiable)
            is_blank = True
            if canvas.json_data is not None:
                objects = canvas.json_data.get("objects", [])
                is_blank = (len(objects) == 0)
            
            # 2) Fallback simple: contar "tinta" (píxeles claros)
            # (por si json_data no viene o viene raro)
            if canvas.image_data is not None and is_blank:
                arr_rgba = canvas.image_data.astype(np.uint8)
                gray = Image.fromarray(arr_rgba).convert("L")
                arr = np.array(gray)
                ink = (arr > 20).sum()   # píxeles claros (el trazo es blanco)
                if ink > 80:             # umbral mínimo de tinta
                    is_blank = False
            
            if canvas.image_data is not None and not is_blank:
                img_arr = canvas.image_data.astype(np.uint8)
                pil_img = Image.fromarray(img_arr).convert("L")
            
            else:
                pil_img = None
                st.info("👆 Dibuja un número en el canvas para predecir.")

    else:
        uploaded = st.file_uploader(
            "Sube una imagen con un dígito (ideal: fondo negro, dígito blanco)",
            type=["png", "jpg", "jpeg"],
        )
        if uploaded is not None:
            pil_img = Image.open(uploaded)

    if pil_img is not None:
        pil_img = pil_img.convert("L")

        # invertimos si es necesario (heurística)
        # if np.mean(np.array(pil_img)) > 127:
        #     pil_img = ImageOps.invert(pil_img)

        st.write("Vista previa (antes de 28×28):")
        st.image(pil_img, use_container_width=True)


with right:
    st.subheader("2) Predicción")

    if pil_img is None:
        st.info("Primero dibuja o sube una imagen.")
    else:
        model, device = load_trained_model(str(WEIGHTS), out_1=16, out_2=32)

        x = preprocess_pil(pil_img)
        st.write("Debug x: mean=", float(x.mean()), "max=", float(x.max()), "sum=", float(x.sum()))

        pred, probs = predict(model, x, device)

        st.markdown(f"## ✅ Predicción: **{pred}**")
        top3 = np.argsort(probs)[::-1][:3]
        st.write("Top-3:", ", ".join([f"{i} ({probs[i]:.2%})" for i in top3]))

        x_img = (x.squeeze(0).squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        st.image(x_img, clamp=True, width=220)

        st.write("Probabilidades por clase:")
        st.bar_chart(probs)
