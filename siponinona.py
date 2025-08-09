import streamlit as st
import pandas as pd
import numpy as np
import mysql.connector
from mysql.connector import Error
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xlsxwriter
from io import BytesIO
import seaborn as sns
import plotly.express as px

# Konfigurasi halaman
st.set_page_config(
    page_title="SIPONINONA",
    page_icon="logo.jpg",
    layout="wide"
)

# Fungsi koneksi database
def get_connection():
    try:
        connection = mysql.connector.connect(
            host="sql12.freesqldatabase.com",     
            user="sql12794167",                 
            password="yhcLMb3BWJ",           
            database="sql12794167",             
            port=3306
        )
        return connection
    except Error as e:
        st.error(f"Error connecting to MySQL: {e}")
        return None

# Fungsi simpan dataset ke database
def save_to_database(df):
    try:
        conn = get_connection()
        if conn:
            cursor = conn.cursor()

            # Bersihkan spasi di nama kolom
            df.columns = df.columns.str.strip()

            # Buat tabel sesuai kolom di file kamu
            create_table_query = """
            CREATE TABLE IF NOT EXISTS dataset (
                no INT PRIMARY KEY,
                nama_kecamatan VARCHAR(255),
                volume_sampah_tidak_terlayani FLOAT,
                jarak_ke_tpa FLOAT,
                jumlah_desa INT,
                jumlah_penduduk INT
            )
            """
            cursor.execute(create_table_query)

            # Hapus data lama
            cursor.execute("TRUNCATE TABLE dataset")

            # Masukkan data
            for _, row in df.iterrows():
                insert_query = """
                INSERT INTO dataset (
                    no, nama_kecamatan, volume_sampah_tidak_terlayani,
                    jarak_ke_tpa, jumlah_desa, jumlah_penduduk
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(insert_query, (
                    int(row['No']),
                    str(row['Nama Kecamatan']),
                    float(row['Volume Sampah Tidak Terlayani']),
                    float(row['Jarak ke TPA']),
                    int(row['Jumlah Desa']),
                    int(row['Jumlah Penduduk'])
                ))

            conn.commit()
            cursor.close()
            conn.close()
            st.success("✅ Data berhasil disimpan ke database!")
    except Exception as e:
        st.error(f"❌ Gagal menyimpan data ke database: {e}")


# Fungsi login
def login_user(username, password):
    conn = get_connection()
    if conn:
        cursor = conn.cursor()
        query = "SELECT * FROM users WHERE username = %s AND password = %s"
        cursor.execute(query, (username, password))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result
    return None

# Inisialisasi session state
if "is_logged_in" not in st.session_state:
    st.session_state.is_logged_in = False
if "df" not in st.session_state:
    st.session_state.df = None
if "df_clustered" not in st.session_state:
    st.session_state.df_clustered = None
if "selected_columns" not in st.session_state:
    st.session_state.selected_columns = None
if "num_clusters" not in st.session_state:
    st.session_state.num_clusters = 3
if "selected_data" not in st.session_state:
    st.session_state.selected_data = None
if "menu" not in st.session_state:
    st.session_state.menu = "🏠 Beranda"
if "df_normalized" not in st.session_state:
    st.session_state.df_normalized = None

# Tampilkan data di setiap halaman
def show_data(df):
    with st.expander("📁 Data yang Sedang Diproses", expanded=True):
        st.dataframe(df)

# Tampilkan kredit di setiap halaman
def show_credit():
    st.markdown("<div style='text-align:center; font-size:12px; color:gray;'>© DLH Kabupaten Bogor 2025</div>", unsafe_allow_html=True)

# Fungsi logout
def logout():
    st.session_state.is_logged_in = False
    st.session_state.user = None
    st.session_state.menu = "🏠 Beranda"

# Tampilkan hanya jika belum login
if not st.session_state.is_logged_in:
    st.markdown("<h1 style='text-align:center;'>SIPONINONA</h1>", unsafe_allow_html=True)
    left, center, right = st.columns([17,9,13])
    with center:
        st.image("logo.jpg", width=130)
    st.markdown("<h3 style='text-align:center;'>Sistem Informasi Pemetaan Kondisi Komponen Pengelolaan Sampah Kabupaten Bogor</h3>", unsafe_allow_html=True)

    left, center, right = st.columns([1, 11, 1])
    with center:
        with st.form("login_form", clear_on_submit=True):
            username = st.text_input("🧑 Username", placeholder= "Masukkan username Anda")
            password = st.text_input("🔑 Password", type="password", placeholder= "Masukkan password Anda")
            login_btn = st.form_submit_button("Login")

        show_credit()

    if login_btn:
        if not username or not password:
            st.warning("Username dan Password tidak boleh kosong!")
        else:
            user = login_user(username, password)
            if user:
                st.session_state.is_logged_in = True
                st.session_state.user = username
                st.session_state.menu = "🏠 Beranda"
                st.rerun()
            else:
                st.error("Username atau password salah!")
else:
    # Sidebar: Navigasi
    with st.sidebar:
        col1, col2 = st.columns([1, 4])
        with col1:
            left, center, right = st.columns([13,9,13])
            st.image("logo.jpg", width=130)
        with col2:
            st.markdown("<div style='font-size:30px; font-weight:bold; margin-top:10px;'>SIPONINONA</div>", unsafe_allow_html=True)
        st.markdown("---")

        menu = st.radio(
            "Navigasi",
            ["🏠 Beranda", "📤 Upload File", "🧮 Hasil Perhitungan", "📊 Diagram Hasil Cluster"],
            label_visibility="collapsed",
            index=["🏠 Beranda", "📤 Upload File", "🧮 Hasil Perhitungan", "📊 Diagram Hasil Cluster"].index(st.session_state.menu)
        )

        st.markdown("<br><br><br><br><br><br>", unsafe_allow_html=True)
        if st.button("Logout"):
            logout()
            st.rerun()

    # Konten Halaman
    if menu == "🏠 Beranda":
        st.markdown("### Selamat datang di SIPONINONA Kabupaten Bogor")
        st.write("""
        **SIPONINONA** adalah platform digital untuk menganalisis kondisi komponen pengelolaan sampah di tiap kecamatan di Kabupaten Bogor.
        """)
        show_credit()

    elif menu == "📤 Upload File":
        st.header("📤 Upload File")
    
        uploaded_file = st.file_uploader("Pilih file Excel atau CSV", type=["xlsx", "csv"])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file, engine='openpyxl')

                st.session_state.df = df
                st.success("File berhasil diupload!")

                # Simpan ke database
                save_to_database(df)

                st.header("Data Awal")
                show_data(df)

                # ... lanjutkan proses normalisasi & konfigurasi clustering seperti di kode kamu ...
                
            except Exception as e:
                st.error(f"Terjadi kesalahan saat membaca file: {str(e)}")

        elif st.session_state.df is not None:
            show_data(st.session_state.df)
            st.info("File sebelumnya masih tersedia.")

