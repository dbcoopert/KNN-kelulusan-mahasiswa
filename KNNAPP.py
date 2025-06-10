# Import library yang dibutuhkan
import streamlit as st  # Untuk membuat UI aplikasi web
import pandas as pd  # Untuk manipulasi dan analisis data
import seaborn as sns  # Untuk visualisasi statistik
import matplotlib.pyplot as plt  # Untuk visualisasi grafik
from sklearn.preprocessing import LabelEncoder, MinMaxScaler  # Untuk encoding dan normalisasi data
from sklearn.neighbors import KNeighborsClassifier  # Algoritma KNN
from sklearn.model_selection import train_test_split  # Membagi data latih dan uji
from sklearn.metrics import classification_report, accuracy_score  # Evaluasi model

# Judul dan deskripsi aplikasi
st.title("🎓 Prediksi Status Kelulusan Mahasiswa")
st.write("Aplikasi ini menggunakan model KNN untuk memprediksi apakah mahasiswa akan lulus tepat waktu atau terlambat.")

# Fungsi untuk memuat dan membersihkan data
@st.cache_data
def load_data():
    df = pd.read_csv("kelulusan_mhs.csv")  # Membaca file CSV
    df = df.drop(columns=["NAMA", "STATUS MAHASISWA"], errors='ignore')  # Hapus kolom tidak diperlukan

    # Isi nilai kosong: rata-rata untuk numerik, modus untuk kategorikal
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

# Memuat data
df = load_data()

# Label Encoding untuk kolom kategorikal
label_cols = ['JENIS KELAMIN', 'STATUS NIKAH', 'STATUS KELULUSAN']
le_dict = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Mengubah label teks ke angka
    le_dict[col] = le  # Simpan encoder untuk decoding nanti

# Normalisasi data numerik ke skala 0-1
num_cols = ['UMUR'] + [f'IPS {i}' for i in range(1, 9)] + ['IPK']
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Pisahkan fitur dan target
X = df.drop(columns=['STATUS KELULUSAN'])
y = df['STATUS KELULUSAN']

# Bagi data latih dan uji (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Slider untuk memilih nilai k
k = st.slider("🔢 Pilih Nilai K (jumlah tetangga)", min_value=1, max_value=15, value=3)
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Evaluasi model
y_pred = knn.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)

st.subheader("📊 Evaluasi Model")
st.write(f"Akurasi: {akurasi:.2f}")

# Classification report dalam bentuk tabel
try:
    target_names = le_dict['STATUS KELULUSAN'].classes_
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
except:
    st.warning("Gagal menampilkan classification report karena masalah label.")

# Visualisasi: Distribusi Status Kelulusan
st.subheader("📈 Distribusi Status Kelulusan")
fig1, ax1 = plt.subplots()
sns.countplot(data=df, x='STATUS KELULUSAN', ax=ax1)
ax1.set_xticklabels(le_dict['STATUS KELULUSAN'].classes_)
ax1.set_ylabel("Jumlah Mahasiswa")
st.pyplot(fig1)

# Visualisasi: Distribusi IPK berdasarkan kelulusan
st.subheader("📊 Distribusi IPK berdasarkan Status Kelulusan")
fig2, ax2 = plt.subplots()
sns.boxplot(data=df, x='STATUS KELULUSAN', y='IPK', ax=ax2)
ax2.set_xticklabels(le_dict['STATUS KELULUSAN'].classes_)
st.pyplot(fig2)

# Visualisasi: Rata-rata IPS per Semester
st.subheader("📉 Rata-rata IPS per Semester")
ips_cols = [f'IPS {i}' for i in range(1, 9)]
mean_ips = df[ips_cols].mean()
fig3, ax3 = plt.subplots()
mean_ips.plot(kind='line', marker='o', ax=ax3)
ax3.set_title("Rata-rata IPS Semester")
ax3.set_xlabel("Semester")
ax3.set_ylabel("IPS (Ternormalisasi)")
st.pyplot(fig3)

# Form Input Mahasiswa Baru
st.subheader("📝 Input Data Mahasiswa Baru")
nama = st.text_input("Nama Mahasiswa")  # Input nama
jenis_kelamin = st.selectbox("Jenis Kelamin", le_dict['JENIS KELAMIN'].classes_)
status_nikah = st.selectbox("Status Nikah", le_dict['STATUS NIKAH'].classes_)
umur = st.slider("Umur", 18, 35, 24)

# Input IPS 1-8
ips_values = []
for i in range(1, 9):
    ips = st.slider(f"IPS Semester {i}", 0.00, 4.00, 3.00, step=0.01)
    ips_values.append(ips)

ipk = st.slider("IPK", 0.00, 4.00, 3.00, step=0.01)

# Buat DataFrame dari input, dan normalisasi
input_data = pd.DataFrame([[umur, *ips_values, ipk]], columns=['UMUR'] + ips_cols + ['IPK'])
input_data[num_cols] = scaler.transform(input_data[num_cols])

# Prediksi
if st.button("🔮 Prediksi Kelulusan"):
    if not nama.strip():
        st.warning("Silakan masukkan nama mahasiswa terlebih dahulu.")
    else:
        # Encode input kategorikal
        encoded_input = pd.DataFrame([[
            le_dict['JENIS KELAMIN'].transform([jenis_kelamin])[0],
            input_data.iloc[0]['UMUR'],
            le_dict['STATUS NIKAH'].transform([status_nikah])[0],
            *input_data.iloc[0][ips_cols],
            input_data.iloc[0]['IPK']
        ]], columns=X.columns)

        hasil_prediksi = knn.predict(encoded_input)[0]
        hasil_label = le_dict['STATUS KELULUSAN'].inverse_transform([hasil_prediksi])[0]

        st.success(f"📌 Prediksi: Mahasiswa akan {hasil_label.upper()} waktu.")

        # Siapkan hasil untuk diunduh
        download_df = pd.DataFrame({
        'Nama': [nama], 
        'Jenis Kelamin': [jenis_kelamin],
        'Status Nikah': [status_nikah],
        'Umur': [umur],
        'IPS 1': [ips_values[0]],
        'IPS 2': [ips_values[1]],
        'IPS 3': [ips_values[2]],
        'IPS 4': [ips_values[3]],
        'IPS 5': [ips_values[4]],
        'IPS 6': [ips_values[5]],
        'IPS 7': [ips_values[6]],
        'IPS 8': [ips_values[7]],
        'IPK': [ipk],
        'Prediksi Kelulusan': [hasil_label.upper()]
    })

        csv = download_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Hasil Prediksi",
            data=csv,
            file_name="hasil_prediksi_mahasiswa.csv",
            mime="text/csv"
        )
