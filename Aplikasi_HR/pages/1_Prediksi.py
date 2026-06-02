import streamlit as st
import numpy as np
import joblib
import cv2
import sys
import os
from PIL import Image

# MEMAKSA COLAB MEMBACA FOLDER FUNCTIONS
if '/content/Aplikasi_HR' not in sys.path:
    sys.path.append('/content/Aplikasi_HR')
from functions.feature_extraction import extract_super_features

# INJEKSI CLASS FDCA
class FDCA:
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.w_a = None; self.w_b = None
    def fit(self, XA, XB, y): pass
    def transform(self, XA, XB): return np.hstack((XA @ self.w_a, XB @ self.w_b))

import __main__
__main__.FDCA = FDCA 

st.set_page_config(page_title="Prediksi - HR Classification", page_icon="🩺", layout="wide")
st.title("🩺 Panel Deteksi Retinopati Hipertensi")
st.write("---")

@st.cache_resource
def load_models():
    model_dir = '/content/drive/MyDrive/Penelitian/Penelitian/'
    scA = joblib.load(os.path.join(model_dir, 'scaler_A.pkl'))
    scB = joblib.load(os.path.join(model_dir, 'scaler_B.pkl'))
    fdca_model = joblib.load(os.path.join(model_dir, 'fdca_model.pkl'))
    gnb_model = joblib.load(os.path.join(model_dir, 'gnb_model.pkl'))
    return scA, scB, fdca_model, gnb_model

try:
    scA, scB, fdca_model, gnb_model = load_models()
    model_loaded = True
except Exception as e:
    st.error(f"⚠️ Model gagal dimuat. Error: {e}")
    model_loaded = False

col_input, col_result = st.columns([1, 1.2], gap="large")

with col_input:
    st.subheader("📥 1. Input Citra Retina")
    uploaded_file = st.file_uploader("Unggah foto fundus retina pasien di sini:", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Preview Citra', use_container_width=True)

with col_result:
    st.subheader("📊 2. Hasil Diagnosis")

    if uploaded_file is not None and model_loaded:
        st.info("Citra siap diproses. Klik tombol di bawah untuk memulai.")

        if st.button('🚀 Mulai Analisis Citra', use_container_width=True, type="primary"):
            with st.spinner('Memproses jaringan pembuluh darah & tekstur (44 Fitur)...'):
                try:
                    img_array = np.array(img.convert('RGB'))
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                    # Ekstraksi Fitur
                    green = img_rgb[:, :, 1]
                    feat_A = extract_super_features(green)

                    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
                    l, a, b = cv2.split(lab)
                    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
                    feat_B = extract_super_features(cl)

                    feat_A = np.array(feat_A).reshape(1, -1)
                    feat_B = np.array(feat_B).reshape(1, -1)

                    # Prediksi
                    feat_A_scaled = scA.transform(feat_A)
                    feat_B_scaled = scB.transform(feat_B)
                    
                    Z_test = fdca_model.transform(feat_A_scaled, feat_B_scaled)
                    prob = gnb_model.predict_proba(Z_test)
                    prediction = np.argmax(prob)

                    st.write("---")
                    if prediction == 1:
                        st.error("### 🚨 TERDETEKSI HIPERTENSI (Hypertensive Retinopathy)")
                    else:
                        st.success("### ✅ NORMAL (Kondisi Retina Sehat)")

                    st.write("")
                    st.markdown("**Detail Probabilitas Model:**")
                    
                    # ---- BAGIAN YANG DIPERBAIKI (MENGHINDARI ERROR st.metric) ----
                    st.info(f"🔴 **Peluang Hipertensi :** {prob[0][1] * 100:.2f} %")
                    st.success(f"🟢 **Peluang Sehat :** {prob[0][0] * 100:.2f} %")
                    # ---------------------------------------------------------------

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
    elif uploaded_file is None:
        st.warning("👈 Silakan unggah gambar di panel kiri terlebih dahulu.")
