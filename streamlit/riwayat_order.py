import streamlit as st
import mysql.connector

# Fungsi untuk membuat koneksi ke database MySQL
def create_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="coffee_db"
    )

# Fungsi untuk mendapatkan riwayat pesanan berdasarkan nama pelanggan
def get_order_history(customer_name):
    conn = create_connection()
    cursor = conn.cursor()
    query = """
    SELECT order_date, coffee_type, quantity, metode_payment
    FROM pick_up_orders
    WHERE customer_name = %s
    ORDER BY order_date DESC
    """
    cursor.execute(query, (customer_name,))
    orders = cursor.fetchall()
    conn.close()
    return orders

# Halaman Streamlit untuk Riwayat Pemesanan
def order_history_page():
    st.header("Riwayat Pemesanan")

    # Input nama pelanggan
    customer_name = st.text_input("Masukkan Nama Anda untuk Melihat Riwayat Pesanan:")
    
    if customer_name:
        # Ambil riwayat pesanan dari database
        orders = get_order_history(customer_name)

        if orders:
            st.subheader(f"Riwayat Pesanan untuk {customer_name}:")
            for order in orders:
                order_date, coffee_type, quantity, metode_payment = order
                st.write(f"- {order_date}: {coffee_type} (Jumlah: {quantity}, Metode Pembayaran: {metode_payment})")
        else:
            st.warning("Tidak ada riwayat pesanan ditemukan untuk nama ini.")
    else:
        st.info("Masukkan nama Anda untuk melihat riwayat pesanan.")
