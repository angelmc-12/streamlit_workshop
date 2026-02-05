"""
Streamlit app: MNIST digit classifier (CNN)

Qué hace esta app:
- Permite dibujar un dígito (0–9) en un canvas o subir una imagen.
- Preprocesa la imagen a formato MNIST (28x28, escala 0–1).
- Carga un modelo CNN entrenado (mnist_cnn.pt) y predice el dígito.
- Muestra probabilidades por clase.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

import torch
import torch.nn as nn


# ============================================================
# 1) Definición del modelo
# ============================================================
class CNN(nn.Module):
    """
    CNN simple para MNIST:
    - Conv -> ReLU -> MaxPool
    - Conv -> ReLU -> MaxPool
    - FC a 10 clases
    """

    def __init__(self, out_1: int = 16, out_2: int = 32):
        super().__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Con input 28x28:
        #  - maxpool1 reduce a 14x14
        #  - maxpool2 reduce a 7x7
        self.fc1 = nn.Linear(out_2 * 7 * 7, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.cnn1(x))
        x = self.maxpool1(x)

        x = torch.relu(self.cnn2(x))
        x = self.maxpool2(x)

        x = x.view(x.size(0), -1)  # flatten
        x = self.fc1(x)
        return x


# ============================================================
# 2) Preprocesamiento y utilidades de inferencia
# ============================================================
def preprocess_pil(pil_img: Image.Image) -> torch.Tensor:
    """
    Convierte una imagen PIL a un tensor MNIST-like:
    - Grayscale
    - Resize 28x28
    - Escala a [0, 1]
    - Devuelve tensor con forma [1, 1, 28, 28]
    """
    img = pil_img.convert("L").resize((28, 28))
    arr = np.array(img).astype(np.float32)# / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def load_trained_model(weights_path: str | Path, out_1: int = 16, out_2: int = 32):
    """
    Carga el modelo y los pesos desde un .pt.

    Nota sobre Streamlit Cloud:
    - Si usas @st.cache_resource, el modelo puede quedarse "pegado" con pesos antiguos.
    - Si quieres cachear, versiona con el mtime o hash del archivo.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(out_1=out_1, out_2=out_2).to(device)

    state = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device


def predict(model: nn.Module, x: torch.Tensor, device: torch.device) -> tuple[int, np.ndarray]:
    """
    Ejecuta inferencia:
    - x: tensor [1,1,28,28]
    Retorna:
    - pred: clase predicha (int)
    - probs: np.array (10,) con probabilidades softmax
    """
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)  # [1,10]
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred = int(np.argmax(probs))
    return pred, probs


def canvas_to_pil_gray(canvas_image_data: np.ndarray) -> Image.Image:
    """
    Convierte la salida del canvas (RGBA) a imagen PIL en escala de grises.

    Por qué existe:
    - streamlit-drawable-canvas devuelve un array RGBA.
    - Convertimos a PIL y a "L" para el preprocesamiento MNIST.
    """
    # streamlit_drawable_canvas entrega normalmente uint8 [0..255]
    arr = canvas_image_data.astype(np.uint8)

    # Convertimos RGBA -> PIL y luego a gris (L)
    # Nota: en la mayoría de casos esto es suficiente.
    return Image.fromarray(arr).convert("L")


# ============================================================
# 3) UI en Streamlit
# ============================================================
st.set_page_config(page_title="MNIST Classifier (Techy)", layout="wide")
st.title("MNIST Classifier — CNN + Demo")
st.caption("Dibuja un dígito o sube una imagen. El modelo predice 0–9 y muestra probabilidades.")

# Ruta del modelo (mismo directorio que app.py)
WEIGHTS = Path(__file__).resolve().parent / "mnist_cnn.pt"
if not WEIGHTS.exists():
    st.error("No encuentro `mnist_cnn.pt` en el repo. Súbelo al mismo nivel que `app.py`.")
    st.stop()

left, right = st.columns([1, 1], gap="large")

# ----------------------------
# 3.1) Entrada: dibujar o subir imagen
# ----------------------------
with left:
    st.subheader("1) Entrada")

    mode = st.radio("Elige modo:", ["🖊️ Dibujar", "🖼️ Subir imagen"], horizontal=True)
    pil_img: Image.Image | None = None

    if mode == "🖊️ Dibujar":
        st.write("Dibuja un dígito (0–9). Ideal: trazo grueso y centrado.")

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

        # 1) Detectar si hay trazos usando json_data (lo más confiable)
        is_blank = True
        if canvas.json_data is not None:
            objects = canvas.json_data.get("objects", [])
            is_blank = (len(objects) == 0)

        # 2) Fallback: contar "tinta" si json_data no refleja trazos
        if canvas.image_data is not None and is_blank:
            gray = Image.fromarray(canvas.image_data.astype(np.uint8)).convert("L")
            arr = np.array(gray)
            ink = (arr > 20).sum()  # píxeles claros (trazo blanco)
            if ink > 80:
                is_blank = False

        # Si hay imagen y no está en blanco, la convertimos a PIL
        if canvas.image_data is not None and not is_blank:
            pil_img = canvas_to_pil_gray(canvas.image_data)
        else:
            st.info("👆 Dibuja un número en el canvas para predecir.")

        # Debug opcional (déjalo en False cuando publiques)
        DEBUG_CANVAS = False
        if DEBUG_CANVAS and canvas.image_data is not None:
            st.write(
                "canvas dtype:", canvas.image_data.dtype,
                "min:", float(np.min(canvas.image_data)),
                "max:", float(np.max(canvas.image_data)),
            )

    else:
        uploaded = st.file_uploader(
            "Sube una imagen con un dígito (ideal: fondo negro, dígito blanco)",
            type=["png", "jpg", "jpeg"],
        )
        if uploaded is not None:
            pil_img = Image.open(uploaded).convert("L")

    # Vista previa
    if pil_img is not None:
        st.write("Vista previa (antes de 28×28):")
        st.image(pil_img, use_container_width=True)


# ----------------------------
# 3.2) Predicción
# ----------------------------
with right:
    st.subheader("2) Predicción")

    if pil_img is None:
        st.info("Primero dibuja o sube una imagen.")
    else:
        # Cargamos el modelo (sin cache para evitar problemas en Cloud)
        model, device = load_trained_model(WEIGHTS, out_1=16, out_2=32)

        # Preprocesamos
        x = preprocess_pil(pil_img)  # [1,1,28,28]

        # Debug del tensor (útil si algo se ve raro)
        st.write("Debug x: mean=", float(x.mean()), "max=", float(x.max()), "sum=", float(x.sum()))

        # Inferencia
        pred, probs = predict(model, x, device)

        # Resultados
        st.markdown(f"## ✅ Predicción: **{pred}**")

        top3 = np.argsort(probs)[::-1][:3]
        st.write("Top-3:", ", ".join([f"{i} ({probs[i]:.2%})" for i in top3]))

        # Mostrar el input final 28x28 tal como lo ve el modelo
        st.write("Input final 28×28 (lo que ve el modelo):")
        x_img = (x.squeeze(0).squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        st.image(x_img, clamp=True, width=220)

        st.write("Probabilidades por clase:")
        st.bar_chart(probs)
