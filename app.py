import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
import joblib
from PIL import Image
import io
import pandas as pd
import seaborn as sns # Untuk visualisasi fitur

# --- KONFIGURASI HALAMAN STREAMLIT ---
# st.set_page_config() harus menjadi perintah Streamlit pertama.
st.set_page_config(
    page_title="Deteksi Pneumonia X-ray",
    layout="wide", # Menggunakan layout lebar untuk visualisasi yang lebih baik
    initial_sidebar_state="expanded" # Sidebar dibuka secara default
)

# --- GLOBAL OBJECTS / INITIALIZATIONS ---
# Inisialisasi objek OpenCV yang dibutuhkan pipeline
# Ini perlu diinisialisasi sekali di awal script Streamlit
clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
kernel = np.ones((3,3),np.uint8)

# --- LOAD MODEL KLASIFIKASI ---
# Pastikan nama file model sesuai dengan yang Anda simpan dari Colab
MODEL_FILENAME = 'svm_pneumonia_detector.pkl' # Sesuaikan jika Anda pakai nama lain
svm_model = None
try:
    svm_model = joblib.load(MODEL_FILENAME)
    st.sidebar.success(f"Model Klasifikasi ({MODEL_FILENAME}) berhasil dimuat!")
except FileNotFoundError:
    st.sidebar.error(f"Error: Model klasifikasi ({MODEL_FILENAME}) tidak ditemukan. Pastikan sudah dilatih dan disimpan di direktori yang sama.")
except Exception as e:
    st.sidebar.error(f"Error saat memuat model: {e}")
    st.sidebar.warning("InconsistentVersionWarning mungkin muncul jika versi scikit-learn berbeda saat menyimpan dan memuat model. Ini dapat diabaikan untuk demo, tetapi disarankan untuk melatih ulang model dengan versi scikit-learn yang sama.")

# --- FUNGSI-FUNGSI PENGOLAHAN CITRA (MIRIP DENGAN COLAB) ---

def process_image_pipeline(img):
    """
    Menjalankan seluruh pipeline pra-pemrosesan dan transformasi pada satu citra.
    """
    # 1. Grayscale Conversion
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Histogram Equalization (CLAHE)
    # clahe_obj sudah diinisialisasi secara global
    clahe_img = clahe_obj.apply(gray_img)

    # 3. Gaussian Blurring
    blurred_img = cv2.GaussianBlur(clahe_img, (5, 5), 0)

    # 4. Thresholding (Otsu's Thresholding)
    _, otsu_thresh = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 5. Morphological Operations (Opening & Closing)
    # kernel sudah diinisialisasi secara global
    img_opening = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    img_final_mask = cv2.morphologyEx(img_opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    return gray_img, clahe_img, blurred_img, otsu_thresh, img_final_mask

def extract_glcm_features_single_image(image, mask):
    """
    Mengekstrak fitur tekstur Haralick (GLCM) dari citra yang di-masking.
    """
    # Terapkan mask ke citra grayscale asli
    masked_img = cv2.bitwise_and(image, image, mask=mask)

    if np.sum(masked_img) == 0:
        return None # Kembalikan None jika tidak ada fitur yang dapat diekstrak (masking menghasilkan area hitam total)

    # Rescale image ke 0-255 jika rentang nilai piksel tidak standar
    if masked_img.max() > 255 or masked_img.min() < 0:
        masked_img = ((masked_img - masked_img.min()) / (masked_img.max() - masked_img.min()) * 255).astype(np.uint8)

    # Pastikan citra 8-bit untuk perhitungan GLCM
    if masked_img.dtype != np.uint8:
        masked_img = masked_img.astype(np.uint8)

    # Hitung GLCM
    # distances = [1] (piksel tetangga terdekat)
    # angles = [0, np.pi/4, np.pi/2, 3*np.pi/4] (0, 45, 90, 135 derajat)
    glcm = graycomatrix(masked_img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)

    # Ekstrak properti dan rata-ratakan dari semua sudut
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    ASM = graycoprops(glcm, 'ASM').mean() # Angular Second Moment

    return [contrast, dissimilarity, homogeneity, energy, correlation, ASM]


# --- TAMPILAN UTAMA APLIKASI STREAMLIT ---
st.title("👨‍🔬 Proyek Tugas Besar PCD: Deteksi Pneumonia dari Citra X-ray")
st.markdown("""
Aplikasi ini mendemonstrasikan *pipeline* pengolahan citra digital dan klasifikasi untuk membantu mendeteksi pneumonia
pada citra X-ray dada.
""")

st.header("Unggah Citra X-ray Anda")
uploaded_file = st.file_uploader(
    "Pilih citra X-ray dada (.jpg, .jpeg, .png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Konversi file yang diunggah ke format OpenCV (numpy array)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR) # Membaca sebagai citra berwarna

    # Tampilkan citra asli
    st.subheader("Citra X-ray Asli")
    st.image(img_original, caption="Citra X-ray yang Diunggah", use_column_width=True, channels="BGR")

    st.markdown("---")

    # Tombol untuk memulai pemrosesan dan klasifikasi
    if st.button("Mulai Proses Analisis dan Deteksi"):
        if svm_model is None:
            st.warning("Model klasifikasi belum dimuat. Tidak dapat melanjutkan proses deteksi.")
        else:
            with st.spinner('Memproses citra dan melakukan prediksi...'):
                # Jalankan pipeline pengolahan citra
                gray_img, clahe_img, blurred_img, otsu_thresh, final_mask = process_image_pipeline(img_original)

                st.subheader("Pipeline Pengolahan Citra Digital")

                # Visualisasi hasil preprocessing dan transformasi
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(gray_img, caption="1. Grayscale Conversion", use_column_width=True, channels='GRAY')
                with col2:
                    st.image(clahe_img, caption="2. CLAHE (Contrast Enhancement)", use_column_width=True, channels='GRAY')
                with col3:
                    st.image(blurred_img, caption="3. Gaussian Blurring (Noise Reduction)", use_column_width=True, channels='GRAY')

                col4, col5 = st.columns(2)
                with col4:
                    st.image(otsu_thresh, caption="4. Otsu's Thresholding (Segmentation)", use_column_width=True, channels='GRAY')
                with col5:
                    st.image(final_mask, caption="5. Morphological Operations (Opening & Closing)", use_column_width=True, channels='GRAY')

                st.markdown("---")

                st.subheader("Analisis dan Ekstraksi Fitur")

                # Visualisasi area yang digunakan untuk ekstraksi fitur (citra grayscale asli yang di-masking)
                original_gray_for_mask = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
                masked_img_for_glcm_display = cv2.bitwise_and(original_gray_for_mask, original_gray_for_mask, mask=final_mask)

                st.image(masked_img_for_glcm_display, caption="Area Paru-paru Tersegmentasi (Digunakan untuk Ekstraksi Fitur)", use_column_width=True, channels='GRAY')
                st.markdown("Area berwarna putih pada gambar di atas adalah bagian yang teridentifikasi sebagai paru-paru dan digunakan untuk ekstraksi fitur tekstur.")

                # Ekstraksi fitur Haralick
                features = extract_glcm_features_single_image(original_gray_for_mask, final_mask)

                if features is not None:
                    features_df = pd.DataFrame([features], columns=['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 'ASM'])
                    st.write("### Fitur Tekstur (Haralick) yang Diekstrak:")
                    st.dataframe(features_df)
                    st.markdown("Fitur-fitur ini mengkuantifikasi karakteristik tekstur citra, seperti kekasaran, kontras, dan homogenitas, yang dapat membedakan kondisi paru-paru normal dari pneumonia.")

                    st.markdown("---")

                    # Prediksi Klasifikasi
                    prediction = svm_model.predict(features_df.values)[0]
                    # Mendapatkan probabilitas untuk setiap kelas
                    prediction_proba = svm_model.predict_proba(features_df.values)


                    # Mendapatkan indeks untuk kelas 'NORMAL' dan 'PNEUMONIA'
                    # Pastikan list(svm_model.classes_) sesuai dengan urutan yang digunakan saat training
                    class_labels = list(svm_model.classes_)
                    proba_normal = 0
                    proba_pneumonia = 0

                    if 'NORMAL' in class_labels:
                        proba_normal = prediction_proba[0][class_labels.index('NORMAL')] * 100
                    if 'PNEUMONIA' in class_labels:
                        proba_pneumonia = prediction_proba[0][class_labels.index('PNEUMONIA')] * 100


                    st.subheader("Hasil Deteksi Pneumonia")
                    if prediction == 'PNEUMONIA':
                        st.error(f"**DIAGNOSIS POTENSIAL: PNEUMONIA** (Keyakinan Model: {proba_pneumonia:.2f}%) ⚠️")
                        st.markdown("Berdasarkan analisis citra X-ray, model mendeteksi adanya indikasi pneumonia.")
                        st.warning("👉🏻 **Penting:** Hasil ini adalah prediksi otomatis. Selalu konsultasikan dengan profesional medis untuk diagnosis dan penanganan yang akurat.")
                    else:
                        st.success(f"**DIAGNOSIS POTENSIAL: NORMAL** (Keyakinan Model: {proba_normal:.2f}%) ✅")
                        st.markdown("Berdasarkan analisis citra X-ray, model tidak mendeteksi adanya indikasi pneumonia.")
                        st.info("👍🏻 **Penting:** Hasil ini adalah prediksi otomatis. Meskipun kondisi paru-paru terlihat normal, selalu disarankan untuk konsultasi medis jika Anda memiliki gejala atau kekhawatiran kesehatan.")

                    st.markdown(f"Detail Probabilitas Klasifikasi: **Normal = {proba_normal:.2f}%**, **Pneumonia = {proba_pneumonia:.2f}%**")

                else:
                    st.warning("Tidak dapat mengekstrak fitur dari citra yang diunggah. Ini mungkin terjadi jika citra terlalu gelap atau area paru-paru tidak dapat disegmentasi dengan jelas.")

# --- SIDEBAR & FOOTER ---
st.sidebar.title("Informasi Proyek")
st.sidebar.markdown("""
**Dosen Pengampu:** Rizky Amelia, S.Si., M.Han.
**Mata Kuliah:** Pengolahan Citra Digital
**Program Studi:** Informatika, Institut Teknologi Kalimantan
""")
st.sidebar.markdown("---")
st.sidebar.info("""
**Catatan:** Aplikasi ini adalah proyek demonstrasi dan bukan alat diagnosis medis.
Hasil prediksi harus dikonfirmasi oleh profesional medis.
""")