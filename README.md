```markdown
# Pneumonia X-Ray Classification

## Deskripsi Proyek
Proyek ini bertujuan untuk mengembangkan aplikasi web berbasis **Streamlit** yang dapat mengklasifikasikan gambar X-Ray dada dan mendeteksi penyakit pneumonia, termasuk jenis pneumonia bakterial, pneumonia viral, dan COVID-19. Aplikasi ini menggunakan dua model deep learning, yaitu **ResNet50V2** dan **VGG16**, yang telah di-fine-tune untuk mengklasifikasikan gambar X-Ray.

Tujuan pengembangan aplikasi ini adalah untuk memberikan alat bantu untuk analisis cepat dan visualisasi hasil diagnosis dengan menggunakan teknologi **Grad-CAM** untuk menyoroti area yang relevan pada gambar X-Ray yang dianggap penting oleh model dalam proses klasifikasi.

## Langkah Instalasi

### 1. Persyaratan Sistem
Pastikan Anda memiliki **Python 3.7+** yang terinstal pada sistem Anda.

### 2. Instalasi Dependencies
Untuk menginstal semua dependencies yang diperlukan, Anda dapat menggunakan `pip`. Ikuti langkah-langkah berikut:

1. Clone repository ini:
    ```bash
    git clone https://github.com/username/pneumonia-xray-classifier.git
    cd pneumonia-xray-classifier
    ```

2. Buat dan aktifkan environment virtual (opsional, tetapi disarankan):
    ```bash
    python -m venv venv
    source venv/bin/activate  # Untuk pengguna Windows, gunakan venv\Scripts\activate
    ```

3. Instal dependencies yang diperlukan:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Menjalankan Aplikasi
Setelah semua dependencies terinstal, Anda bisa menjalankan aplikasi dengan perintah berikut:
```bash
streamlit run app.py
```

Aplikasi akan berjalan pada [http://localhost:8501](http://localhost:8501).

## Deskripsi Model
Model yang digunakan dalam aplikasi ini adalah **ResNet50V2** dan **VGG16**, yang merupakan dua arsitektur deep learning terkenal dalam tugas klasifikasi gambar.

### **ResNet50V2**
Model ini menggunakan **ResNet50V2**, sebuah convolutional neural network (CNN) yang dilatih untuk mengklasifikasikan gambar X-Ray ke dalam empat kategori: Normal, Pneumonia-Bacterial, Pneumonia-Viral, dan COVID-19. Model ini menggunakan residual connections yang membantu dalam pelatihan jaringan yang lebih dalam.

### **VGG16**
Model kedua adalah **VGG16**, arsitektur CNN yang lebih sederhana dibandingkan ResNet, tetapi tetap efektif dalam banyak tugas klasifikasi gambar. VGG16 menggunakan lapisan konvolusi yang lebih kecil dan lebih banyak lapisan dibandingkan model lainnya.

Kedua model ini dilatih dengan dataset X-Ray dada yang mencakup gambar dari pasien dengan kondisi berbeda. Setiap model dilatih untuk mengenali pola yang dapat membedakan kondisi tersebut.

## Hasil dan Analisis

### Hasil Evaluasi Model

#### **ResNet50V2**
Berikut adalah hasil klasifikasi dari model ResNet50V2:

| Metrik              | Normal | Pneumonia-Bacterial | Pneumonia-Viral | COVID-19 |
|---------------------|--------|---------------------|-----------------|----------|
| **Precision**       | 0.90   | 0.77                | 0.65            | 0.98     |
| **Recall**          | 0.99   | 0.82                | 0.46            | 0.96     |
| **F1-Score**        | 0.94   | 0.80                | 0.54            | 0.97     |
| **Accuracy**        | 0.84   |                     |                 |          |
| **Macro Avg**       | 0.83   | 0.81                | 0.81            |          |
| **Weighted Avg**    | 0.82   | 0.84                | 0.83            |          |

#### **VGG16**
Berikut adalah hasil klasifikasi dari model VGG16:

| Metrik              | Normal | Pneumonia-Bacterial | Pneumonia-Viral | COVID-19 |
|---------------------|--------|---------------------|-----------------|----------|
| **Precision**       | 0.89   | 0.76                | 0.65            | 0.98     |
| **Recall**          | 0.99   | 0.82                | 0.42            | 0.96     |
| **F1-Score**        | 0.94   | 0.79                | 0.51            | 0.97     |
| **Accuracy**        | 0.83   |                     |                 |          |
| **Macro Avg**       | 0.82   | 0.80                | 0.80            |          |
| **Weighted Avg**    | 0.82   | 0.83                | 0.82            |          |

### Analisis Perbandingan
Kedua model, **ResNet50V2** dan **VGG16**, menunjukkan performa yang baik dengan akurasi masing-masing **84%** dan **83%**. ResNet50V2 sedikit lebih unggul dalam hal akurasi dan F1-score, terutama dalam mengklasifikasikan gambar **Pneumonia-Viral**. VGG16 menunjukkan sedikit kelemahan dalam pengklasifikasian gambar **Pneumonia-Viral**, yang berhubungan dengan recall yang lebih rendah pada kategori tersebut.

### Visualisasi Grad-CAM
Untuk meningkatkan interpretabilitas, aplikasi ini juga menggunakan **Grad-CAM** untuk menyoroti area gambar X-Ray yang relevan dalam klasifikasi. Area yang lebih terang menunjukkan pentingnya bagi model dalam membuat keputusan klasifikasi.

## Link Live Demo
Anda dapat mengakses aplikasi web yang telah di-deploy di [tautan ini](https://example-link-to-deployed-app.com). Aplikasi ini memungkinkan pengguna untuk meng-upload gambar X-Ray dan mendapatkan analisis diagnosis serta visualisasi hasilnya.

---

Terima kasih telah menggunakan aplikasi ini! Kami berharap aplikasi ini dapat memberikan bantuan dalam analisis gambar X-Ray untuk mendeteksi pneumonia dan COVID-19.
```

Jangan lupa untuk mengganti `https://example-link-to-deployed-app.com` dengan URL aktual aplikasi yang telah di-deploy.
