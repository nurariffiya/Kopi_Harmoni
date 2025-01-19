import streamlit as st
import pandas as pd
from koneksi import create_connection

def coffee_menu_page():
    st.subheader("Data Transaksi Coffee")

    # Create database connection
    connection = create_connection()
    cursor = connection.cursor()

    # Fetch all data from the 'menu_coffee' table
    query = "SELECT * FROM menu_coffee"
    cursor.execute(query)
    data = cursor.fetchall()
    
    # Convert the data into a DataFrame for display
    df = pd.DataFrame(data, columns=['id', 'tanggal_transaksi', 'customer', 'item', 'kopi (g)', 
                                      'susu (g)', 'evaporasi (g)', 'sauce_caramel (g)', 'cafein (mg)', 'calories'])
    connection.close()

    # Display the data in a table
    st.write("Transaksi Coffee:")
    st.write(df)


