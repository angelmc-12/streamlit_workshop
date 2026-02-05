import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image

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

with right:
    st.subheader("2) Predicción")
