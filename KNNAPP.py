import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Judul Aplikasi
st.title("🎓 Prediksi Status Kelulusan Mahasiswa")
st.write("Aplikasi ini menggunakan model KNN untuk memprediksi apakah mahasiswa akan lulus tepat waktu atau terlambat.")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("kelulusan_mhs.csv")
    df = df.drop(columns=["NAMA", "STATUS MAHASISWA"], errors='ignore')
    
    # Isi nilai kosong dengan rata-rata
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

df = load_data()

# Label Encoding
label_cols = ['JENIS KELAMIN', 'STATUS NIKAH', 'STATUS KELULUSAN']
le_dict = {}

for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Normalisasi fitur numerik
num_cols = ['UMUR'] + [f'IPS {i}' for i in range(1, 9)] + ['IPK']
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Fitur dan Target
X = df.drop(columns=['STATUS KELULUSAN'])
y = df['STATUS KELULUSAN']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model KNN
k = st.slider("🔢 Pilih Nilai K (jumlah tetangga)", min_value=1, max_value=15, value=3)
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Evaluasi Model
y_pred = knn.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)

st.subheader("📊 Evaluasi Model")
st.write(f"Akurasi: {akurasi:.2f}")

try:
    target_names = le_dict['STATUS KELULUSAN'].classes_
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
except:
    st.warning("Gagal menampilkan classification report karena masalah label.")

# Visualisasi 1: Distribusi Status Kelulusan
st.subheader("📈 Distribusi Status Kelulusan")
fig1, ax1 = plt.subplots()
sns.countplot(data=df, x='STATUS KELULUSAN', ax=ax1)
ax1.set_xticklabels(le_dict['STATUS KELULUSAN'].classes_)
ax1.set_ylabel("Jumlah Mahasiswa")
st.pyplot(fig1)

# Visualisasi 2: Distribusi IPK berdasarkan Status Kelulusan
st.subheader("📊 Distribusi IPK berdasarkan Status Kelulusan")
fig2, ax2 = plt.subplots()
sns.boxplot(data=df, x='STATUS KELULUSAN', y='IPK', ax=ax2)
ax2.set_xticklabels(le_dict['STATUS KELULUSAN'].classes_)
st.pyplot(fig2)

# Visualisasi 3: Rata-rata IPS per Semester
st.subheader("📉 Rata-rata IPS per Semester")
ips_cols = [f'IPS {i}' for i in range(1, 9)]
mean_ips = df[ips_cols].mean()
fig3, ax3 = plt.subplots()
mean_ips.plot(kind='line', marker='o', ax=ax3)
ax3.set_title("Rata-rata IPS Semester")
ax3.set_xlabel("Semester")
ax3.set_ylabel("IPS (Ternormalisasi)")
st.pyplot(fig3)

# Form Input Data Mahasiswa Baru
st.subheader("📝 Input Data Mahasiswa Baru")
jenis_kelamin = st.selectbox("Jenis Kelamin", le_dict['JENIS KELAMIN'].classes_)
status_nikah = st.selectbox("Status Nikah", le_dict['STATUS NIKAH'].classes_)
umur = st.slider("Umur", 18, 35, 24)

ips_values = []
for i in range(1, 9):
    ips = st.slider(f"IPS Semester {i}", 0.00, 4.00, 3.00, step=0.01)
    ips_values.append(ips)

ipk = st.slider("IPK", 0.00, 4.00, 3.00, step=0.01)

# Normalisasi input baru
input_data = pd.DataFrame([[
    umur, *ips_values, ipk
]], columns=['UMUR'] + ips_cols + ['IPK'])
input_data[num_cols] = scaler.transform(input_data[num_cols])

# Prediksi
if st.button("🔮 Prediksi Kelulusan"):
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

