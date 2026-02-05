import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image,ImageOps
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


st.set_page_config(page_title="MNIST Classifier (Techy)", layout="wide")

st.title("MNIST Classifier — From Scratch (CNN) + Demo")
st.caption("Dibuja un dígito o sube una imagen. El modelo predice 0–9 y muestra probabilidades.")

st.code(f"""
import streamlit as st
""")

left, right = st.columns([1, 1])

with left:
    st.subheader("1) Entrada")
    mode = st.radio("Elige modo:", ["🖊️ Dibujar", "🖼️ Subir imagen"], horizontal=True)

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

        img_arr = canvas.image_data.astype(np.uint8)
        pil_img = Image.fromarray(img_arr).convert("L")

    
    else:
        uploaded = st.file_uploader("Sube una imagen con un dígito (ideal fondo negro, dígito blanco)", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            pil_img = Image.open(uploaded)

    if pil_img is not None:
        pil_img = pil_img.convert("L")
        if np.mean(np.array(pil_img)) > 127:
            pil_img = ImageOps.invert(pil_img)
            
        st.write("Vista previa (antes de 28x28):")
        st.image(pil_img, use_container_width=True)


with right:
    st.subheader("2) Predicción")
