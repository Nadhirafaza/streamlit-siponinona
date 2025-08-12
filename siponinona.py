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
from sklearn.metrics import silhouette_samples, silhouette_score

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

        # Sidebar: Navigasi
        menu = st.radio(
            "Navigasi",
            ["🏠 Beranda", "📤 Upload File", "🧮 Hasil Perhitungan", "📊 Diagram Hasil Cluster", "📈 Evaluasi Hasil"],
            label_visibility="collapsed",
            index=["🏠 Beranda", "📤 Upload File", "🧮 Hasil Perhitungan", "📊 Diagram Hasil Cluster", "📈 Evaluasi Hasil"].index(st.session_state.menu)
        )

        st.markdown("<br><br><br><br><br><br>", unsafe_allow_html=True)
        if st.button("Logout"):
            logout()
            st.rerun()

    # Konten Halaman
    if menu == "🏠 Beranda":
        st.markdown("### Selamat datang di SIPONINONA Kabupaten Bogor")
        st.write("""
        **SIPONINONA (Sistem Informasi Pemetaan Kondisi Komponen Pengelolaan Sampah)** adalah platform digital berbasis data yang dirancang untuk:
        
        🔍 **Menganalisis kondisi komponen pengelolaan sampah** di tiap kecamatan di Kabupaten Bogor  
        📊 **Mengelompokkan wilayah berdasarkan kondisi komponen pengelolaan sampah** menggunakan metode *K-Means*  
        🧠 **Membantu perencanaan strategis dan alokasi sumber daya** oleh pemerintah daerah dan stakeholder terkait
        
        ---
        ### 🚀 Cara Penggunaan Sistem:
        1. Upload file data pengelolaan sampah 
        2. Masukan variabel, nilai awal centroid, dan tentukan jumlah cluster yang diinginkan
        3. Lihat hasil perhitungan dan visualisasi data dalam bentuk diagram
        
        💡 *Pastikan data Anda lengkap, terutama kolom numerik seperti: volume sampah, jarak ke TPA, jumlah desa, dan jumlah penduduk.*
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

                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                
                if not numeric_cols:
                    st.error("File tidak mengandung kolom numerik.")
                else:
                    # Normalisasi data menggunakan decimal scaling
                    df_normalized = df.copy()
                    columns_to_normalize = [col for col in numeric_cols if col.lower() != "no"]  # Hindari kolom "No"

                    for col in columns_to_normalize:
                        max_value = df_normalized[col].max()
                        if max_value != 0:
                            df_normalized[col] = df_normalized[col] / max_value

                    st.session_state.df_normalized = df_normalized
                    st.session_state.selected_columns = numeric_cols

                    st.header("Data Setelah Normalisasi")
                    show_data(df_normalized)
                    
                    st.header("Konfigurasi Clustering")
                    cols1, cols2 = st.columns(2)
                    
                    with cols1:
                    # Filter agar kolom "No" tidak muncul di pilihan
                        numeric_cols_no = [col for col in numeric_cols if col.lower() != "no"]

                    selected_columns = st.multiselect(
                        "Pilih variabel untuk clustering",
                        numeric_cols_no,
                        default=numeric_cols_no[:2] if len(numeric_cols_no) >= 2 else numeric_cols_no,
                        key="cols_selector"
                    )
                    st.session_state.selected_columns = selected_columns

                    with cols2:
                        num_clusters = st.slider(
                            "Jumlah cluster (k)",
                            2, 10, 3,
                            key="cluster_slider"
                        )
                        st.session_state.num_clusters = num_clusters
                    
                    jumlah_data = len(df_normalized)
                    centroid_positions = []
                    for i in range(num_clusters):
                        posisi = int(jumlah_data / (i + 2))
                        centroid_positions.append(posisi)

                    st.markdown("### Nilai Centroid Awal")
                    st.info(f"Jumlah data: {df_normalized.shape[0]}")
                    st.info("Rumus Menentukan Nilai Centroid Awal: **(n Data) / (n Cluster + 1)**")

                    for i, posisi in enumerate(centroid_positions):
                        st.markdown(f"- **C{i+1}** = {jumlah_data} / ({i+1}+1) = {jumlah_data // (i+2)} → Nilai centroid awal diambil dari baris ke-**{posisi}** menjadi baris data ke-**{posisi-1}**")

                    if selected_columns:
                        st.markdown("### Memilih Nilai Centroid Awal")
                        sample_df = df_normalized[selected_columns].copy()
                        st.info("Pilih baris data untuk nilai centroid awal sesuai saran posisi diatas.")

                        centroid_cols = st.columns(num_clusters)
                        selected_data = []

                        for i in range(num_clusters):
                            with centroid_cols[i]:
                                st.markdown(f"### Centroid Awal {i+1}")
                                row_idx = st.selectbox(
                                    f"Pilih baris untuk Centroid {i+1}",
                                    options=sample_df.index,
                                    format_func=lambda x: f"Baris {x}",
                                    key=f"centroid_{i}"
                                )
                                selected_values = df_normalized.loc[row_idx, selected_columns].values
                                st.write(dict(zip(selected_columns, selected_values)))
                                selected_data.append(selected_values)

                        st.session_state.selected_data = selected_data
                        st.success("Konfigurasi disimpan!")

                        show_credit()

            except Exception as e:
                st.error(f"Terjadi kesalahan saat membaca file: {str(e)}")

        elif st.session_state.df is not None:
            show_data(st.session_state.df)
            st.info("File sebelumnya masih tersedia. Upload file baru jika ingin mengganti.")


    elif menu == "🧮 Hasil Perhitungan":
        st.header("🧮 Proses Clustering")
        
        if st.session_state.df_normalized is not None:
            show_data(st.session_state.df_normalized)
            
        try:
            if st.button("🚀 Jalankan Clustering"):
                df = st.session_state.df_normalized
                selected_columns = st.session_state.selected_columns
                selected_data = st.session_state.selected_data
                num_clusters = st.session_state.num_clusters

                # Konversi ke array
                X = df[selected_columns].values
                initial_centroids = np.array(selected_data)

                kmeans = KMeans(
                    n_clusters=num_clusters,
                    init=initial_centroids,
                    n_init=1
                )
                clusters = kmeans.fit_predict(X)

                df_clustered = df.copy()
                df_clustered['Cluster'] = clusters + 1
                st.session_state.df_clustered = df_clustered

                st.success("Clustering berhasil!")

                st.subheader("Hasil Clustering")
                st.dataframe(df_clustered.sort_values("Cluster"))

                st.subheader("Nilai Centroid Akhir")
                st.write(df_clustered.groupby("Cluster")[selected_columns].mean())
                
                show_credit()
                
                if len(selected_columns) == 2:
                    st.subheader("Visualisasi Cluster")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', edgecolor='k', alpha=0.7)
                    ax.scatter(initial_centroids[:, 0], initial_centroids[:, 1], c='red', s=200, marker='*', label='Centroid Awal', edgecolor='k')
                    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='blue', s=200, marker='X', label='Centroid Akhir', edgecolor='k')
                    ax.set_xlabel(selected_columns[0])
                    ax.set_ylabel(selected_columns[1])
                    ax.set_title("Visualisasi K-Means Clustering")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"Terjadi error: {str(e)}")


    elif menu == "📊 Diagram Hasil Cluster":
        st.header("📊 Visualisasi Cluster")
        
        if st.session_state.df_clustered is not None:
            show_data(st.session_state.df_clustered)
            
            df_clustered = st.session_state.df_clustered
            selected_columns = st.session_state.selected_columns
            X = df_clustered[selected_columns].values

            # Add a download button for the clustered DataFrame as Excel
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df_clustered.to_excel(writer, sheet_name='Clustered Data', index=False)
            excel_buffer.seek(0)

            st.download_button(
                label="📥 Download Data sebagai Excel",
                data=excel_buffer,
                file_name="data_clustered.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
           
            # Mapping cluster ke label deskriptif
            cluster_label_map = {
                1: '1 TPS3R',
                2: '2 Bank Sampah',
                3: '3 Armada'
            }

           # Diagram Pie: Persentase tiap cluster
            cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
            fig_pie = px.pie(
                names=cluster_counts.index.map(cluster_label_map),
                values=cluster_counts.values,
                title="Persentase Tiap Cluster",
                hole=0.4,
                color=cluster_counts.index.map(cluster_label_map),
                color_discrete_map={
                    '1 TPS3R': 'orange',
                    '2 Bank Sampah': 'blue',
                    '3 Armada': 'green'
                }
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
                
            # PCA dan visualisasi interaktif
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)

            df_pca = pd.DataFrame(data=X_pca, columns=['Dim1', 'Dim2'])
            df_clustered['Cluster'] = df_clustered['Cluster'].astype(str)  # Pastikan bertipe string
            df_pca['Nama Kecamatan'] = df_clustered['Nama Kecamatan']

            # Langkah 4: Mapping label cluster ke label deskriptif
            cluster_label_map = {
                '1': '1 TPS3R',
                '2': '2 Bank Sampah',
                '3': '3 Armada'
            }
            df_pca['Cluster'] = df_clustered['Cluster'].map(cluster_label_map)

            # Langkah 5: Plot visualisasi dengan plotly
            fig = px.scatter(
                df_pca,
                x='Dim1',
                y='Dim2',
                color='Cluster',         # Warna berdasarkan nama cluster baru
                symbol='Cluster',        # Simbol juga berbeda per cluster
                hover_name='Nama Kecamatan',
                title='Visualisasi Clustering',
                labels={
                    'Dim1': f"Dim1 ({round(pca.explained_variance_ratio_[0]*100, 1)}%)",
                    'Dim2': f"Dim2 ({round(pca.explained_variance_ratio_[1]*100, 1)}%)"
                },
                width=800,
                height=600,
                category_orders={  # ✅ Atur urutan legend
                'Cluster': ['1 TPS3R', '2 Bank Sampah', '3 Armada']
                },
                color_discrete_map={
                    '1 TPS3R': 'orange',
                    '2 Bank Sampah': 'blue',
                    '3 Armada': 'green'
                },
                symbol_map={
                    '1 TPS3R': 'circle',
                    '2 Bank Sampah': 'square',
                    '3 Armada': 'diamond'
                }
            )
            
            fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
            fig.update_layout(legend_title_text='Cluster', hovermode='closest')

            # Tampilkan plot interaktif
            st.plotly_chart(fig, use_container_width=True)
            
            show_credit()

        else:
            st.warning("Silakan lakukan clustering terlebih dahulu di menu Hasil Perhitungan")

    elif menu == "📈 Evaluasi Hasil":
     st.header("📈 Evaluasi Hasil Clustering")


        # === 1. Baca file Excel ===
    file_path = "data sillhouette.xlsx"  # pastikan file ada di folder yang sama
    df = pd.read_excel(file_path, sheet_name="Sheet2", header=1)

        # === 2. Kolom yang dipakai ===
    numeric_cols = ["Vol Sampah", "Jarak", "JLH Desa", "JLH Pend"]
    cluster_col = "Kelompok"

        # === 3. Bersihkan data ===
    df = df.dropna(subset=numeric_cols + [cluster_col])
    for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Konversi label cluster jadi angka
    df["Cluster"] = df[cluster_col].astype(str).str.extract(r"(\d+)").astype(int)

        # === 4. Siapkan data untuk perhitungan silhouette ===
    X = df[numeric_cols].values
    labels = df["Cluster"].values

        # === 5. Hitung silhouette score ===
    sil_samples = silhouette_samples(X, labels, metric="euclidean")
    sil_avg = silhouette_score(X, labels, metric="euclidean")

    df["Silhouette"] = sil_samples

        # === 6. Tampilkan hasil di terminal ===
    print("📊 Rata-rata Silhouette Score:", sil_avg)
    print("\n📋 Contoh nilai silhouette per data:")
    print(df[[cluster_col, "Silhouette"]].head())

        # === 7. Simpan hasil ke Excel ===
    output_file = "hasil_silhouette.xlsx"
    df.to_excel(output_file, index=False)
    print(f"\n✅ Hasil lengkap disimpan ke {output_file}")

        # === 8. Visualisasi distribusi silhouette score ===
    plt.figure(figsize=(8, 5))
    plt.hist(sil_samples, bins=20, color="skyblue", edgecolor="black")
    plt.axvline(sil_avg, color="red", linestyle="--", label=f"Rata-rata = {sil_avg:.4f}")
    plt.title("Distribusi Nilai Silhouette")
    plt.xlabel("Silhouette Coefficient")
    plt.ylabel("Frekuensi")
    plt.legend()
    plt.tight_layout()
    plt.show()

        # === 9. Boxplot per cluster ===
    plt.figure(figsize=(8, 5))
    df.boxplot(column="Silhouette", by="Cluster", grid=False)
    plt.title("Distribusi Silhouette per Cluster")
    plt.suptitle("")  # hapus judul otomatis pandas
    plt.xlabel("Cluster")
    plt.ylabel("Silhouette Coefficient")
    plt.tight_layout()
    plt.show()

