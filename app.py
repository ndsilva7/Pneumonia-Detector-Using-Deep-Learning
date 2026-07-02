import streamlit as st
from keras.models import load_model
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Pneumonia Classifier", page_icon="🫁")

# CSS para aumentar as letras
st.markdown("""
<style>
    p, div, label { font-size: 20px !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='font-size: 48px'>🫁 Pneumonia Classifier</h1>", unsafe_allow_html=True)

@st.cache_resource
def load():
    model = load_model("keras_model.h5", compile=False)  # ← 4 espaços
    return model                                          # ← 4 espaços

model = load()
class_names = ['Pneumonia', 'Normal']

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    div[data-testid="stFileUploader"] { margin-top: -30px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<p style='font-size: 24px !important; margin-bottom: 0px;'>🩻 Send an X-Ray</p>", unsafe_allow_html=True)
uploaded = st.file_uploader("", type=["jpg","jpeg","png"])

if uploaded is not None:
    pil_image = Image.open(uploaded).convert("RGB")  # ← 4 espaços
    st.image(pil_image, caption="Image uploaded")    # ← 4 espaços

    img = np.array(pil_image)
    image = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence = prediction[0][index]

    st.write(f"**- Class:** {class_name}")
    st.write(f"**- Confidence:** {np.round(confidence * 100, 2)}%")