Berikut adalah penjelasan lengkap mengenai proyek, langkah instalasi, model yang digunakan, hasil dan analisis, serta link live demo yang dapat dimasukkan dalam file `README.md`:

---

## Deskripsi Proyek

Proyek ini bertujuan untuk mengembangkan aplikasi web berbasis **Streamlit** yang memungkinkan pengguna untuk mengklasifikasikan gambar X-Ray dada dan mendeteksi jenis-jenis pneumonia serta COVID-19. Aplikasi ini memanfaatkan dua model deep learning yang telah dilatih sebelumnya, yaitu **ResNet50V2** dan **VGG16**, yang digunakan untuk menganalisis gambar X-Ray dan memberikan diagnosis yang akurat berdasarkan kondisi yang terdeteksi pada gambar.

### Latar Belakang
Pneumonia dan COVID-19 merupakan penyakit pernapasan yang dapat didiagnosis melalui pemeriksaan X-Ray dada. Penggunaan model deep learning seperti **ResNet50V2** dan **VGG16** memungkinkan untuk menganalisis gambar X-Ray dengan lebih cepat dan efisien, sehingga membantu tenaga medis dalam memberikan diagnosis yang lebih tepat dan lebih cepat.

### Tujuan Pengembangan
Aplikasi ini bertujuan untuk:
- Mengklasifikasikan gambar X-Ray dada menjadi empat kategori: **Normal**, **Pneumonia-Bacterial**, **Pneumonia-Viral**, dan **COVID-19**.
- Menyediakan hasil klasifikasi beserta analisis menggunakan visualisasi **Grad-CAM**, yang menunjukkan area gambar X-Ray yang dianggap penting oleh model dalam membuat keputusan klasifikasi.
- Memberikan akses mudah bagi pengguna untuk meng-upload gambar X-Ray dan mendapatkan diagnosis dengan cepat.

---

## Langkah Instalasi

Untuk menjalankan aplikasi web ini, berikut adalah langkah-langkah yang perlu dilakukan:

### 1. Persyaratan Sistem
Pastikan sistem Anda telah terinstal **Python 3.7** atau versi yang lebih baru.

### 2. Instalasi Dependencies
1. **Clone repository**:
   ```bash
   git clone https://github.com/dimasdzakiad/pneumonia-classification.git
   cd pneumonia-classification
   ```

2. **Buat dan aktifkan environment virtual** (opsional, tapi disarankan):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Untuk Windows gunakan venv\Scripts\activate
   ```

3. **Instal dependencies**:
   Install semua dependencies yang diperlukan dengan menjalankan:
   ```bash
   pip install -r requirements.txt
   ```

4. **Menjalankan aplikasi**:
   Setelah semua dependencies terinstal, jalankan aplikasi dengan perintah berikut:
   ```bash
   streamlit run pneumonia-classification.py
   ```
   Aplikasi akan berjalan pada [http://localhost:8501](http://localhost:8501).

---

## Deskripsi Model

Proyek ini menggunakan dua model deep learning, yaitu **ResNet50V2** dan **VGG16**, yang telah dioptimalkan untuk tugas klasifikasi gambar X-Ray.

### **ResNet50V2**
ResNet50V2 adalah model Convolutional Neural Network (CNN) yang memiliki keunggulan dalam mengatasi masalah degradasi pada jaringan yang sangat dalam, berkat **residual connections**. Model ini sangat baik dalam mengidentifikasi pola-pola penting pada gambar X-Ray untuk membedakan antara kategori-kategori seperti **Normal**, **Pneumonia-Bacterial**, **Pneumonia-Viral**, dan **COVID-19**.

**Performanya**:
- **Akurasinya** mencapai **84%**.
- Menggunakan **F1-score**, **precision**, dan **recall** sebagai metrik utama evaluasi.
- Performa terbaik ditemukan pada kategori **Normal** dan **COVID-19**, tetapi memiliki performa yang lebih rendah pada kategori **Pneumonia-Viral**.

### **VGG16**
VGG16 adalah model CNN yang lebih sederhana dibandingkan ResNet50V2, namun masih efektif dalam tugas klasifikasi gambar. VGG16 menggunakan lebih banyak lapisan konvolusi dengan ukuran filter yang lebih kecil dan lebih banyak lapisan dalam arsitekturnya. 

**Performanya**:
- **Akurasinya** mencapai **83%**.
- VGG16 juga menampilkan hasil yang baik pada kategori **Normal** dan **COVID-19**, tetapi memiliki performa lebih rendah pada **Pneumonia-Viral** dibandingkan dengan ResNet50V2.

Kedua model ini dilatih menggunakan dataset gambar X-Ray dada untuk mendeteksi penyakit pneumonia dan COVID-19, serta untuk mempelajari pola-pola visual yang membedakan setiap kategori.

---

## Hasil dan Analisis

### Hasil Evaluasi Model

#### **Hasil Evaluasi Model ResNet50V2**:
- **Precision, Recall, dan F1-Score** untuk setiap kelas adalah sebagai berikut:

| Metrik             | Normal | Pneumonia-Bacterial | Pneumonia-Viral | COVID-19 |
|--------------------|--------|---------------------|-----------------|----------|
| **Precision**      | 0.90   | 0.77                | 0.65            | 0.98     |
| **Recall**         | 0.99   | 0.82                | 0.46            | 0.96     |
| **F1-Score**       | 0.94   | 0.80                | 0.54            | 0.97     |
| **Accuracy**       | 0.84   |                     |                 |          |

#### **Hasil Evaluasi Model VGG16**:
- **Precision, Recall, dan F1-Score** untuk setiap kelas adalah sebagai berikut:

| Metrik             | Normal | Pneumonia-Bacterial | Pneumonia-Viral | COVID-19 |
|--------------------|--------|---------------------|-----------------|----------|
| **Precision**      | 0.89   | 0.76                | 0.65            | 0.98     |
| **Recall**         | 0.99   | 0.82                | 0.42            | 0.96     |
| **F1-Score**       | 0.94   | 0.79                | 0.51            | 0.97     |
| **Accuracy**       | 0.83   |                     |                 |          |

### **Perbandingan Performansi Model**
- **ResNet50V2** memberikan hasil yang sedikit lebih baik dibandingkan **VGG16**, terutama dalam hal **Pneumonia-Viral**, di mana ResNet memiliki recall yang lebih tinggi.
- Secara keseluruhan, **ResNet50V2** menghasilkan akurasi 84%, sedangkan **VGG16** menghasilkan akurasi 83%.

### **Visualisasi Grad-CAM**
- Aplikasi ini juga dilengkapi dengan visualisasi **Grad-CAM**, yang menyoroti area pada gambar X-Ray yang dianggap penting oleh model dalam membuat keputusan. Hal ini membantu dalam interpretasi hasil klasifikasi dan memberikan wawasan tentang bagaimana model bekerja.

---

## Link Live Demo

Aplikasi web yang telah di-deploy dapat diakses melalui tautan berikut:
[https://example-link-to-deployed-app.com](https:......)

Tautan ini akan membawa Anda ke aplikasi Streamlit yang memungkinkan Anda untuk meng-upload gambar X-Ray dan menerima hasil klasifikasi yang cepat dan akurat, lengkap dengan visualisasi **Grad-CAM**.

---

Terima kasih telah menggunakan aplikasi ini! Kami berharap aplikasi ini dapat membantu dalam analisis gambar X-Ray untuk mendeteksi pneumonia dan COVID-19 dengan lebih efisien dan akurat.

---

Dengan penjelasan ini, Anda dapat memberikan gambaran lengkap mengenai aplikasi dan cara menjalankannya. Pastikan untuk mengganti link live demo dengan URL yang valid setelah aplikasi di-deploy.
