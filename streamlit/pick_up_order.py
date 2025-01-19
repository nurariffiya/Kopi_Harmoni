import mysql.connector
import streamlit as st
from datetime import datetime
import pandas as pd

# Fungsi untuk membuat koneksi ke database MySQL
def create_connection():
    return mysql.connector.connect(
        host="localhost",         # Alamat server MySQL
        user="root",              # Username default XAMPP
        password="",              # Password default XAMPP
        database="coffee_db"      # Nama database Anda
    )

# Fungsi untuk menambahkan pesanan ke database
def add_order(customer_name, coffee_type, size, sugar_level, ice_level, quantity, notes, payment_method):
    conn = create_connection()
    cursor = conn.cursor()
    query = """
    INSERT INTO pick_up_orders (customer_name, coffee_type, size, sugar_level, ice_level, quantity, notes, metode_payment, order_date)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    order_date = datetime.now()  # Waktu pemesanan saat ini
    cursor.execute(query, (customer_name, coffee_type, size, sugar_level, ice_level, quantity, notes, payment_method, order_date))
    conn.commit()
    conn.close()

# Fungsi untuk mengambil data menu kopi dari tabel rekomendasi
def get_recommended_menu():
    conn = create_connection()
    query = "SELECT * FROM rekomendasi"  # Menampilkan semua data di tabel rekomendasi
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Halaman Streamlit untuk Pick-Up Order
def pick_up_order_page():
    st.header("Pick-Up Order")
    
    # Mengambil data menu dari database
    menu_df = get_recommended_menu()

    # Menampilkan menu lengkap untuk referensi
    st.subheader("Menu Kopi")
    st.write(menu_df)

    # Dropdown untuk memilih kategori kalori
    st.subheader("Pilih Kategori Kalori")
    calorie_categories = menu_df['kategori_kalori'].unique()
    selected_category = st.selectbox("Kategori Kalori:", calorie_categories)

    # Filter menu berdasarkan kategori kalori yang dipilih
    filtered_menu = menu_df[menu_df['kategori_kalori'] == selected_category]

    # Jika kategori dipilih, tampilkan menu yang sesuai
    st.subheader("Menu Berdasarkan Kategori")
    st.write(filtered_menu[['nama', 'kategori_kalori']])  # Menampilkan kolom yang relevan saja

    with st.form("pickup_form"):
        # Nama pelanggan
        customer_name = st.text_input("Nama Pemesan:", placeholder="Masukkan nama Anda")

        # Pilihan jenis kopi (berdasarkan kategori kalori yang dipilih)
        coffee_type = st.selectbox("Pilih Kopi:", filtered_menu['nama'].tolist())

        # Pilihan ukuran
        size = st.selectbox("Pilih Ukuran:", ["Medium"])

        # Pilihan tingkat gula
        sugar_level = st.radio("Pilih Tingkat Gula:", ["Normal", "Less Sugar", "No Sugar"])

        # Pilihan tingkat es
        ice_level = st.radio("Pilih Tingkat Es:", ["Normal", "Less Ice", "No Ice"])

        # Jumlah pesanan
        quantity = st.number_input("Jumlah:", min_value=1, step=1, value=1)

        # Catatan tambahan (opsional, teks pendek)
        notes = st.text_input("Catatan Tambahan (Opsional):", placeholder="Misalnya: Tanpa susu")

        # Metode pembayaran (hanya tunai)
        payment_method = "Tunai"  # Fixed to "Tunai"

        submitted = st.form_submit_button("Pesan")
        if submitted:
            add_order(customer_name, coffee_type, size, sugar_level, ice_level, quantity, notes, payment_method)
            st.success("Pesanan Anda berhasil dibuat! Silakan ambil di tempat.")
            st.write("Detail Pesanan:")
            st.write(f"Nama: {customer_name}")
            st.write(f"Kopi: {coffee_type}")
            st.write(f"Ukuran: {size}")
            st.write(f"Tingkat Gula: {sugar_level}")
            st.write(f"Tingkat Es: {ice_level}")
            st.write(f"Jumlah: {quantity}")
            st.write(f"Catatan: {notes if notes else 'Tidak ada'}")
            st.write(f"Metode Pembayaran: {payment_method}")

# Jalankan halaman pick-up order
if __name__ == "__main__":
    pick_up_order_page()
