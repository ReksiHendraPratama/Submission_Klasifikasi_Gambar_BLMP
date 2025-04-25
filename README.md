# Proyek Klasifikasi Gambar Hewan 🐶🕷️🐎

## 📝 Ringkasan

Proyek ini bertujuan untuk membangun model klasifikasi gambar hewan menggunakan pendekatan *Convolutional Neural Network* (CNN) dengan teknik *transfer learning*. Klasifikasi gambar hewan adalah salah satu aplikasi kunci dalam *computer vision*, dan proyek ini dirancang untuk mengenali tiga jenis hewan—**anjing**, **laba-laba**, dan **kuda**—berdasarkan citra digital. Dengan memanfaatkan model pre-trained **MobileNetV2**, pelatihan dilakukan secara efisien sambil mencapai akurasi tinggi (>95%). Proyek ini dibangun menggunakan Python dengan pustaka *deep learning* seperti TensorFlow dan Keras, dan seluruh proses pelatihan serta evaluasi dilakukan di **Google Colab** dengan akselerasi GPU T4.

Proyek ini merupakan bagian dari *submission* untuk **Kelas Belajar Pengembangan Machine Learning Dicoding** oleh **Reksi Hendra Pratama**. Mari kita jelajahi bagaimana model ini mencapai prediksi dengan tingkat kepercayaan luar biasa: **99,99% untuk anjing**, **98,08% untuk kuda**, dan **100,0% untuk laba-laba**! 🎉

---

## 📚 Library

Proyek ini menggunakan berbagai pustaka Python untuk mendukung pengolahan data, pembangunan model, evaluasi, dan konversi model. Berikut adalah pustaka utama yang digunakan:

- **TensorFlow (2.18.0)**: *Framework* utama untuk membangun, melatih, dan mengevaluasi model CNN.
- **Keras**: Antarmuka tingkat tinggi untuk merancang arsitektur model dengan mudah.
- **tensorflowjs**: Untuk mengkonversi model ke format TensorFlow.js agar dapat digunakan di web.
- **NumPy**: Untuk manipulasi array dan operasi numerik.
- **Pandas**: Untuk pengolahan data tabular (jika diperlukan).
- **Matplotlib**: Untuk visualisasi plot akurasi, *loss*, dan *confusion matrix*.
- **Seaborn**: Untuk visualisasi *confusion matrix* yang lebih menarik.
- **Scikit-learn**: Untuk menghitung metrik evaluasi seperti *classification report* dan *confusion matrix*.
- **PIL (Pillow)**: Untuk pemrosesan dan manipulasi gambar.
- **os, zipfile, shutil, pathlib**: Untuk manipulasi file dan direktori.
- **google.colab**: Untuk integrasi dengan Google Colab, termasuk unggah file dan penyimpanan ke Google Drive.

Dependensi lengkap dapat dilihat di file `requirements.txt`, yang dihasilkan dengan perintah:
```bash
!pip freeze > requirements.txt
```

---

## 📂 Pembagian Data

Dataset **Animal-10** (diunduh dari [Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10)) berisi **12.307 gambar** hewan, yang dibagi menjadi tiga bagian:
- **Training Set**: ~8.615 gambar (70%) untuk melatih model.
- **Validation Set**: ~2.462 gambar (20%) untuk memantau performa selama pelatihan.
- **Test Set**: ~1.230 gambar (10%) untuk evaluasi akhir pada data yang belum pernah dilihat.

**Augmentasi data** diterapkan pada *training set* menggunakan `ImageDataGenerator` untuk meningkatkan keragaman citra dan mencegah *overfitting*. Teknik augmentasi meliputi:
- Rotasi acak (hingga 20 derajat).
- *Horizontal flip*.
- *Zoom* acak.
- Pergeseran piksel (lebar dan tinggi).
- Normalisasi piksel (rescaling ke rentang 0-1).


---

## 🧠 Model

Model dibangun menggunakan arsitektur **Sequential** dengan mengadopsi *transfer learning* dari **MobileNetV2** sebagai *feature extractor*. Beberapa lapisan awal MobileNetV2 dibekukan untuk mempertahankan bobot pre-trained, sementara lapisan lainnya di-*fine-tune* untuk menyesuaikan dengan dataset. Lapisan tambahan ditambahkan untuk klasifikasi:
- **GlobalAveragePooling2D**: Mengurangi dimensi fitur.
- **Dropout (0.5)**: Mencegah *overfitting*.
- **Dense**: Lapisan keluaran dengan aktivasi *softmax* untuk tiga kelas (anjing, laba-laba, kuda).

**Konfigurasi Pelatihan**:
- **Loss Function**: *Categorical Crossentropy* untuk klasifikasi multi-kelas.
- **Metrik**: Akurasi.

**Callback** digunakan untuk optimasi pelatihan:
- **EarlyStopping**: Menghentikan pelatihan jika `val_accuracy` tidak meningkat setelah 5 epoch (dengan *baseline* 96%).
- **ModelCheckpoint**: Menyimpan model terbaik berdasarkan `val_loss` ke `best_model.keras`.
- **ReduceLROnPlateau**: Mengurangi *learning rate* sebesar 50% jika `val_accuracy` stagnan selama 3 epoch.

---

## **💡 Cara Menjalankan**  
### **1️⃣ Install Dependensi**  
Pastikan Anda memiliki Python dan semua library yang diperlukan dengan menjalankan perintah berikut:  
```bash  
pip install -r requirements.txt  
```  

### **2️⃣ Jalankan Model untuk Inferensi**  
Gunakan model yang telah dilatih untuk melakukan prediksi pada gambar baru:  
```bash  
python predict.py --image_path "images/sample.jpg"  
```  

### **3️⃣ Konversi Model ke Format Lain**  
#### **Konversi ke TFLite**  
```bash  
python convert_to_tflite.py  
```  
#### **Konversi ke TensorFlow.js**  
```bash  
tensorflowjs_converter --input_format=keras model.h5 tfjs_model  
```  

---

## 📊 Hasil Evaluasi

Model menunjukkan performa luar biasa dengan **akurasi >95%** pada data validasi dan uji, jauh melampaui syarat minimum 85%. Hasil inferensi pada gambar uji menunjukkan tingkat kepercayaan yang sangat tinggi:
- 🐶 **Anjing**: 99,99%
- 🐎 **Kuda**: 98,08%
- 🕷️ **Laba-laba**: 100,0%

**Evaluasi** dilakukan dengan:
- **Confusion Matrix**: Menunjukkan prediksi seimbang tanpa bias antar kelas.
- **Classification Report**: Presisi, *recall*, dan F1-score tinggi untuk semua kelas.
- **Plot Akurasi dan Loss**: Menunjukkan konvergensi model yang stabil tanpa tanda *overfitting*.

Hasil ini membuktikan bahwa model mampu menggeneralisasi dengan baik pada data baru, menjadikannya solusi yang andal untuk klasifikasi gambar hewan.

---

## ✅ Kesimpulan

Dengan memanfaatkan *transfer learning* dari MobileNetV2 dan augmentasi data yang efektif, model CNN dalam proyek ini berhasil mengklasifikasikan gambar hewan dengan akurasi sangat tinggi (>95%). Hasil evaluasi menunjukkan bahwa model tidak hanya akurat, tetapi juga mampu menggeneralisasi dengan baik pada data yang belum pernah dilihat. Proyek ini menunjukkan kekuatan *transfer learning* dalam menyelesaikan tugas klasifikasi citra secara efisien, bahkan dengan dataset yang kompleks seperti Animal-10.

Model ini dapat menjadi fondasi untuk pengembangan sistem klasifikasi gambar hewan lebih lanjut atau aplikasi *computer vision* di domain lain, seperti deteksi spesies atau monitoring satwa liar.

---

## 🙋 Tentang Saya

- **Nama**: Reksi Hendra Pratama
- **Email**: reksihendrapratama637@gmail.com
- **ID Dicoding**: MC189D5Y0840

Proyek ini adalah bukti komitmen saya untuk menguasai *machine learning* dan *computer vision*. Saya berharap proyek ini dapat menginspirasi dan memberikan wawasan bagi siapa saja yang tertarik dengan klasifikasi gambar! 😊

---



## 📬 Kontak dan Kontribusi

Ingin berdiskusi atau berkontribusi? Hubungi saya di [reksihendrapratama637@gmail.com](mailto:reksihendrapratama637@gmail.com). Semua masukan sangat dihargai! 🙌

Terima kasih telah menjelajahi proyek ini. Mari kita terus belajar dan menciptakan solusi *machine learning* yang luar biasa! 🚀
