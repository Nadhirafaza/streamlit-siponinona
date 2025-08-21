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
from scipy.spatial.distance import cdist

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
            user="sql12795318",                 
            password="1fVUMDJ15n",           
            database="sql12795318",             
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
        
        🔍 **Menganalisis kondisi komponen pengelolaan sampah** di tiap kecamatan di Kabupaten Bogor.  
        📊 **Mengelompokkan wilayah berdasarkan kondisi komponen pengelolaan sampah** menggunakan metode *K-Means*.  
        🧠 **Membantu perencanaan strategis dan alokasi sumber daya** oleh pemerintah daerah dan stakeholder terkait.
        
        ---
        ### 🚀 Cara Penggunaan Sistem:
        1. Upload file data pengelolaan sampah.
        2. Masukan variabel, nilai awal centroid, dan tentukan jumlah cluster yang diinginkan.
        3. Lihat hasil perhitungan dan visualisasi data dalam bentuk diagram.
        
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
                    st.markdown("""
                    ###   💡 Penjelasan
                    Pada tahap **Konfigurasi Clustering**, pengguna diminta untuk memilih variabel yang relevan agar hasil pengelompokan sesuai dengan kondisi lapangan.  
                    
                    Variabel yang dipilih yaitu:  

                    - **Volume Sampah Tidak Terlayani** → menunjukkan besarnya permasalahan sampah yang belum tertangani.  
                    - **Jarak ke TPA** → memengaruhi efisiensi pengangkutan sampah dan kebutuhan fasilitas tambahan.  
                    - **Jumlah Desa** → mencerminkan luas serta kompleksitas wilayah yang dilayani.  
                    - **Jumlah Penduduk** → semakin banyak penduduk, semakin tinggi potensi timbulan sampah.  

                    Jumlah cluster ditentukan **k = 3** karena penelitian ini diarahkan pada **3 jenis rekomendasi alokasi fasilitas pengelolaan sampah**:
                    1. **TPS3R** (Tempat pengolahan sampah Reduce-Reuse-Recycle).  
                    2. **Bank Sampah** (Fasilitas daur ulang sampah).  
                    3. **Armada** (Kendaraan angkut sampah menuju TPA).  

                    Baris untuk nilai centroid awal dipilih sesuai rumus agar cluster yang terbentuk lebih mudah dianalisis dan memudahkan rekomendasi alokasi fasilitas pengelolaan sampah.        

                    Dengan konfigurasi ini, setiap wilayah dapat dikelompokkan sesuai kebutuhan utama pengelolaan sampah, sehingga hasil analisis lebih **praktis dan sistematis**.
                    """)

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

            # Baca file koordinat dari folder proyek
            coords_df = pd.read_excel("data_koordinat.xlsx")  # pastikan ada kolom Nama Kecamatan, Latitude, Longitude
            df_map = pd.merge(df_clustered, coords_df, on='Nama Kecamatan', how='left')

            df_map['Cluster'] = df_map['Cluster'].astype(int)

            # Buat kolom Cluster_Label berdasarkan mapping
            cluster_label_map = {
            1: '1 TPS3R',
            2: '2 Bank Sampah',
            3: '3 Armada'
            }
            
            symbol_map = {
            '1 TPS3R': 'circle',
            '2 Bank Sampah': 'square',
            '3 Armada': 'diamond'
            }

            df_map['Marker_Symbol'] = df_map['Cluster_Label'].map(symbol_map)
            
            fig_map = px.scatter_mapbox(
            df_map,
            lat="Latitude",
            lon="Longitude",
            hover_name="Nama Kecamatan",
            hover_data=["Cluster_Label", "Volume Sampah Tidak Terlayani"],
            color="Cluster_Label",
            symbol="Marker_Symbol",
            zoom=10,
            height=600,
            category_orders={'Cluster_Label': ['1 TPS3R', '2 Bank Sampah', '3 Armada']},
            color_discrete_map={'1 TPS3R':'orange','2 Bank Sampah':'blue','3 Armada':'green'},
            symbol_sequence=['circle', 'square', 'diamond']  # urutan simbol
            )

            fig_map.update_layout(
                mapbox_style="carto-positron",
                margin={"r":0,"t":0,"l":0,"b":0},
                legend_title_text='Cluster'
            )

            st.plotly_chart(fig_map, use_container_width=True)

            # Catatan interpretasi tiap cluster
            st.markdown("### 📌 Catatan Interpretasi Cluster")
            st.info("""
                - **Cluster 1 (1 TPS3R)**: Wilayah ini disarankan untuk alokasi fasilitas TPS3R.  
                - **Cluster 2 (2 Bank Sampah)**: Wilayah ini disarankan untuk alokasi fasilitas Bank Sampah.  
                - **Cluster 3 (3 Armada)**: Wilayah ini disarankan untuk alokasi fasilitas Armada.
                      
                💡 Catatan: Penentuan cluster ini berdasarkan hasil clustering K-Means dan sebaiknya dikombinasikan dengan data lapangan dan evaluasi rutin.
                """)
            show_credit()

        else:
            st.warning("⚠️ Silakan lakukan clustering terlebih dahulu di menu Hasil Perhitungan")

    elif menu == "📈 Evaluasi Hasil":
        st.header("Silhouette Score")

        if st.session_state.df_clustered is not None:

            df_clustered = st.session_state.df_clustered.copy()
            selected_columns = st.session_state.selected_columns
            X = df_clustered[selected_columns].values
            labels = df_clustered['Cluster'].astype(int).values

            # Hitung matriks jarak Euclidean antar semua titik
            dist_matrix = cdist(X, X, metric='euclidean')

            silhouette_values = []
            for i in range(len(X)):
                same_cluster_idx = np.where(labels == labels[i])[0]
                same_cluster_idx = same_cluster_idx[same_cluster_idx != i]  # buang dirinya sendiri
                if len(same_cluster_idx) > 0:
                    a_i = np.mean(dist_matrix[i, same_cluster_idx])
                else:
                    a_i = 0

                b_i_list = []
                for cluster_id in np.unique(labels):
                    if cluster_id != labels[i]:
                        other_cluster_idx = np.where(labels == cluster_id)[0]
                        b_i_list.append(np.mean(dist_matrix[i, other_cluster_idx]))
                b_i = min(b_i_list)

                if max(a_i, b_i) > 0:
                    s_i = (b_i - a_i) / max(a_i, b_i)
                else:
                    s_i = 0
                silhouette_values.append(s_i)

            # Nilai rata-rata silhouette
            silhouette_avg = np.mean(silhouette_values)
            st.success(f"Nilai Silhouette Coefficient: **{silhouette_avg:.4f}**")

            # Simpan ke dataframe
            df_clustered['S(i)'] = silhouette_values
            st.dataframe(df_clustered)

            # === Plot grafik silhouette per titik ===
            fig, ax = plt.subplots()
            ax.bar(range(len(silhouette_values)), silhouette_values, color='orange')
            ax.axhline(y=silhouette_avg, color='red', linestyle='--', label=f'Rata-rata: {silhouette_avg:.4f}')
            ax.set_xlabel("Index Data")
            ax.set_ylabel("Silhouette Coefficient")
            ax.set_title("Nilai Silhouette per Titik Data")
            ax.legend()
            st.pyplot(fig)

            if silhouette_avg < 0.50:
                st.info("""
                    ⚠️ **Catatan:**  

                    1. Hasil clustering ini tetap berguna sebagai panduan awal untuk pengelompokan wilayah.
                    2. Informasi ini dapat membantu alokasi fasilitas pengelolaan sampah agar lebih tepat sasaran.
                    3. Hasil clustering dapat dikombinasikan dengan data lapangan dan evaluasi rutin agar pengelolaan sampah menjadi lebih efektif.
                    """)

            show_credit()

        else:
            st.warning("⚠️ Silakan lakukan clustering terlebih dahulu di menu Hasil Perhitungan.")
