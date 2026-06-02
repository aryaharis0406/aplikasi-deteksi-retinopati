import streamlit as st
import numpy as np

# OBAT ANTI ERROR JOBLIB
class FDCA:
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.w_a = None; self.w_b = None
    def fit(self, XA, XB, y): pass
    def transform(self, XA, XB): return np.hstack((XA @ self.w_a, XB @ self.w_b))

st.set_page_config(page_title="Beranda - HR Classification", page_icon="👁️", layout="wide")

st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>Aplikasi Prediksi Hypertensive Retinopathy</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #64748B;'>Menggunakan Pendekatan Naive Bayes & FDCA</h3>", unsafe_allow_html=True)
st.write("---")

col1, col2 = st.columns([1.5, 1], gap="large")

with col1:
    st.subheader("🔬 Tentang Aplikasi")
    st.info("Aplikasi ini memetakan fitur fisik pada citra retina ke dalam properti numerik untuk mendeteksi penyakit Retinopati Hipertensi (HR).")
    st.markdown("""
    **Metode Ekstraksi 44 Fitur Klinis:**
    * 🩸 **Vessel Geometry**: *Frangi Filter* & *Canny Edge Detection* pada Green Channel.
    * 🦠 **Texture & Lesion**: *Blob Detection*, *GLCM*, & *LBP* pada L-channel (CLAHE).
    """)

with col2:
    st.subheader("👨‍💻 Tim Peneliti")
    st.success("""
    1. **Rizwan Arisandi, S.Si., M.Si.**
    2. **Arya Haris Prasetio Siahaan**
    3. **Dimas Elang Setyoko, S.Kom., M.Cs.**
    4. **Adhe Lingga Dewi, S.Si., M.Si.**
    5. **Agri Adriel Bororing, S.Kom., M.Kom.**
    6. **Brian Nicholas Tedjo**
    7. **Puspita Kartikasari, S.Si., M.Si.**
    """)
