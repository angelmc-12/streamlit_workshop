import streamlit as st

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

    else:
        uploaded = st.file_uploader("Sube una imagen con un dígito (ideal fondo negro, dígito blanco)", type=["png", "jpg", "jpeg"])

with right:
    st.subheader("2) Predicción")
