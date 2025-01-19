import streamlit as st
import mysql.connector
from koneksi import create_connection

def login_page():
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = authenticate_user(username, password)
        if user:
            st.success(f"Welcome {user[1]}")  # user[1] adalah nama pengguna
            st.session_state["logged_in"] = True
            st.session_state["role"] = "barista"  # Gunakan "barista" sebagai peran tetap
        else:
            st.error("Invalid username or password")

def authenticate_user(username, password):
    connection = create_connection()
    cursor = connection.cursor()
    query = "SELECT * FROM login WHERE username=%s AND password=%s"
    cursor.execute(query, (username, password))
    user = cursor.fetchone()
    connection.close()
    return user
