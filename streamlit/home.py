import streamlit as st
import pandas as pd
import mysql.connector

# Fungsi untuk membuat koneksi ke database MySQL
def create_connection():
    return mysql.connector.connect(
        host="localhost",         # Alamat server MySQL
        user="root",              # Username default XAMPP
        password="",              # Password default XAMPP
        database="coffee_db"      # Nama database Anda
    )

# Fungsi untuk mengambil data dari database
def get_data_from_database():
    conn = create_connection()
    query = "SELECT nama, kalori, kategori_kalori FROM rekomendasi"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Fungsi utama untuk menampilkan konten di Streamlit
def home_page():
    st.title("Kopi Rendah dan Tinggi Kalori: Pilihan untuk Gaya Hidup Modern")

    st.write("""
        Di era modern ini, kopi bukan hanya minuman, tetapi juga bagian dari gaya hidup. 
        Dengan semakin banyaknya kesadaran akan pentingnya pola makan sehat, konsumen kini mencari alternatif kopi 
        yang sesuai dengan kebutuhan kalori mereka. Kami menyediakan daftar kopi berdasarkan kategori kalori rendah dan tinggi, 
        agar Anda dapat menikmati kopi favorit tanpa khawatir berlebihan terhadap asupan kalori.
    """)

    st.write("""
        **Apa itu Kopi Rendah Kalori?**  
        Kopi rendah kalori adalah pilihan yang tepat bagi mereka yang sedang diet atau ingin menjaga berat badan. 
        Biasanya, kopi ini mengandung sedikit atau tanpa gula, serta bahan tambahan rendah kalori lainnya.

        **Apa itu Kopi Tinggi Kalori?**  
        Kopi tinggi kalori memberikan pengalaman rasa yang lebih kaya, sering kali mengandung susu, krim, atau gula tambahan. 
        Cocok untuk momen-momen spesial atau jika Anda ingin menikmati secangkir kopi yang lebih memanjakan.
    """)

    # Mengambil data dari database
    try:
        df = get_data_from_database()
    except Exception as e:
        st.error("Gagal mengambil data dari database. Pastikan koneksi dan tabel database sudah benar.")
        st.error(f"Error: {e}")
        return

    # Menampilkan data menu kopi awal
    st.subheader("Daftar Menu Kopi")
    st.write("""
        Berikut adalah daftar kopi yang telah dikelompokkan berdasarkan kandungan kalorinya:
    """)
    st.dataframe(df)
    
# Menjalankan aplikasi Streamlit
if __name__ == "__main__":
    home_page()
