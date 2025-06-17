# pneumonia-xray-detection-pcd

# Deteksi Pneumonia dari Citra X-ray Dada Menggunakan Pengolahan Citra Digital dan Machine Learning

## Pendahuluan

Proyek ini bertujuan untuk mengembangkan sistem otomatis yang dapat mendeteksi pneumonia dari citra X-ray dada menggunakan teknik pengolahan citra digital (PCD) dan algoritma *machine learning*. Diagnosis pneumonia yang cepat dan akurat sangat krusial, terutama di tengah kondisi global saat ini. Dengan memanfaatkan kekuatan pengolahan citra dan pembelajaran mesin, kami berupaya menyediakan alat bantu yang dapat mendukung diagnosis klinis.

## Gambaran Umum Proyek

* **Tujuan Utama:** Mengklasifikasikan citra X-ray dada secara otomatis sebagai "NORMAL" atau "PNEUMONIA".
* **Pendekatan:** Implementasi *pipeline* PCD komprehensif mulai dari pra-pemrosesan citra, segmentasi, ekstraksi fitur tekstur, hingga klasifikasi menggunakan model *machine learning*.
* **Dataset:** Menggunakan dataset publik X-ray dada yang berisi citra normal dan pneumonia.

## Fitur Utama

* **Pra-pemrosesan Citra:** Meningkatkan kualitas citra X-ray yang seringkali memiliki kontras rendah dan *noise*.
* **Segmentasi Paru-paru:** Mengidentifikasi dan mengisolasi area paru-paru dari citra X-ray.
* **Ekstraksi Fitur Tekstur:** Mengkuantifikasi pola tekstur unik yang membedakan paru-paru normal dan terinfeksi pneumonia.
* **Klasifikasi Otomatis:** Menggunakan model *machine learning* untuk memprediksi keberadaan pneumonia.
* **Antarmuka Pengguna Sederhana (Streamlit):** Memungkinkan pengguna mengunggah citra X-ray dan melihat hasil analisis secara interaktif.

## Dataset

Proyek ini menggunakan **Chest X-ray Dataset (Pneumonia)** yang tersedia di Kaggle. Dataset ini terdiri dari citra X-ray dada yang dikategorikan sebagai "NORMAL" atau "PNEUMONIA", dibagi menjadi set *training*, *validation*, dan *test*.

**Link Dataset:** [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## Pipeline Pengolahan Citra Digital

Proyek ini menerapkan minimal 5 teknik PCD, yang terbagi dalam tiga tahap utama:

1.  **Pra-pemrosesan dan Peningkatan Kualitas Citra:**
    * **Grayscale Conversion:** Mengubah citra ke skala abu-abu.
    * **Histogram Equalization (CLAHE):** Meningkatkan kontras lokal citra.
    * **Gaussian Blurring:** Mengurangi *noise* pada citra.
2.  **Pemrosesan dan Transformasi:**
    * **Otsu's Thresholding:** Melakukan segmentasi awal untuk memisahkan objek dari latar belakang.
    * **Morphological Operations (Opening & Closing):** Membersihkan dan menghaluskan area yang tersegmentasi.
3.  **Analisis dan Ekstraksi Fitur:**
    * **Haralick Features (GLCM - Gray Level Co-occurrence Matrix):** Mengekstrak fitur tekstur dari area paru-paru yang tersegmentasi untuk kuantifikasi tekstur jaringan.

## Klasifikasi

Model **Support Vector Machine (SVM)** digunakan untuk mengklasifikasikan citra X-ray dada berdasarkan fitur Haralick yang diekstraksi. Model dilatih pada dataset X-ray dada yang telah diproses untuk membedakan antara kondisi paru-paru normal dan pneumonia.

## Instalasi

Untuk menjalankan proyek ini secara lokal, ikuti langkah-langkah berikut:

1.  **Clone repositori ini:**
    ```bash
    git clone [https://github.com/dionp3/pneumonia-xray-detection-pcd.git](https://github.com/dionp3/pneumonia-xray-detection-pcd.git)
    cd pneumonia-xray-detection-pcd
    ```

2.  **Buat dan aktifkan *virtual environment*:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Di Linux/macOS
    # atau `.\venv\Scripts\activate` # Di Windows
    ```

3.  **Instal dependensi:**
    ```bash
    pip install -r requirements.txt
    ```
    *Catatan: Pastikan Anda telah mengunduh dataset X-ray dari Kaggle dan meletakkannya di lokasi yang sesuai, atau sesuaikan path di kode Colab Anda.*

4.  **Unduh model yang sudah dilatih:**
    Model SVM yang sudah dilatih (`svm_pneumonia_detector.pkl`) harus diletakkan di *root* direktori proyek (`pneumonia-xray-detection-pcd/`). Anda bisa mendapatkannya dari *output* Google Colab *notebook* Anda.

## Cara Menjalankan Aplikasi

Setelah semua dependensi terinstal dan model tersedia, jalankan aplikasi Streamlit:

```bash
streamlit run app.py
```
Aplikasi akan terbuka di browser Anda (biasanya http://localhost:8501).

Thank You
