import mysql.connector

# Fungsi untuk membuat koneksi ke database MySQL
def create_connection():
    return mysql.connector.connect(
        host="localhost",         # Alamat server MySQL (localhost jika menggunakan XAMPP)
        user="root",              # Username default XAMPP
        password="",              # Password default XAMPP (kosongkan jika tidak ada password)
        database="coffee_db"   # Ganti dengan nama database yang Anda buat
    )