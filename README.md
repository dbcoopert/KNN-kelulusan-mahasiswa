# 🎓 Prediksi Status Kelulusan Mahasiswa
Aplikasi berbasis web untuk memprediksi apakah seorang mahasiswa akan *lulus tepat waktu atau terlambat* menggunakan algoritma K-Nearest Neighbors (KNN). Aplikasi ini dibuat menggunakan Python dan Streamlit.

## 📌 Deskripsi Singkat
Proyek ini bertujuan untuk membantu pihak akademik dalam memantau potensi kelulusan mahasiswa berdasarkan data akademik seperti IPS tiap semester, IPK, umur, status nikah, dan jenis kelamin. Model prediksi dibangun menggunakan pendekatan supervised learning (KNN), dilatih dari dataset kelulusan mahasiswa.

## 🚀 Fitur Aplikasi
- Input data mahasiswa baru secara interaktif
- Prediksi status kelulusan: *Tepat* atau *Terlambat*
- Visualisasi:
  - Distribusi kelulusan
  - Boxplot IPK
  - Grafik rata-rata IPS per semester
- Download hasil prediksi sebagai file CSV

## 🧠 Teknologi & Tools
- Python 3.x
- Streamlit
- Scikit-learn
- Pandas
- Matplotlib & Seaborn

## 🧼 Preprocessing Data
- Nilai kosong diisi menggunakan *rata-rata* (untuk numerik) dan *modus* (untuk kategorikal)
- Data kategorikal diubah ke numerik menggunakan *LabelEncoder*
- Fitur numerik seperti IPS, IPK, dan umur dinormalisasi menggunakan *Min-Max Scaling* ke rentang [0–1]

## 🧪 Model
- Algoritma: *K-Nearest Neighbors*
- Parameter k dapat dipilih oleh pengguna
- Data dibagi 80% training dan 20% testing
- Akurasi model: *83%* pada data uji

## 📂 Cara Menjalankan Aplikasi
1. Pastikan Python sudah terinstal.
2. Install dependensi:
   ```bash
   pip install -r requirements.txt
   streamlit run KNNAPP.py
   ```

## 🏞 Tampilan di Streamlit
[Streamlit](https://knn-kelulusan-mahasiswa.streamlit.app/)

## 📃 Lisensi
  Proyek ini dibuat untuk keperluan edukasi dan tugas UAS. Bebas digunakan dan dimodifikasi.

## 💻 Kontributor
- Kelompok 3 Machine Learning TI23H:
  1. Asep Mardianto
  2. Yulinda Fitri
  3. Fajar Andriansyah
  4. Saka Langit Pratita Susilo
  5. Andrean Saputra
     
