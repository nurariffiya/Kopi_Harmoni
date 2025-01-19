import streamlit as st
from streamlit_option_menu import option_menu
from login import login_page
from menu_kalori import menu_kalori_page
from coffee_menu import coffee_menu_page
from model_algoritma import model_algorithm_page
from home import home_page
from pick_up_order import pick_up_order_page
from riwayat_order import order_history_page


# Fungsi utama
def main():
    # Judul aplikasi
    st.markdown(
        """
        <style>
        .main-title {
            text-align: center;
            font-size: 50px;
            font-family: 'Verdana', sans-serif;
            color: #2E2E2E;  /* Dark gray */
        }
        .sub-title {
            text-align: center;
            font-size: 18px;
            font-family: 'Verdana', sans-serif;
            color: #5A5A5A;  /* Medium gray */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='main-title'>Kopi Harmoni Bintara</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Rasakan Harmoni dalam Setiap Seruputan</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Open Daily 13:00 - 23:00 (Senin Closed)</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Hanya menerima pembayaran Tunai</div>", unsafe_allow_html=True)
    
    # Gaya sidebar
    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-color: #D1CFCF;  /* Light gray */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Inisialisasi session state
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    # Sidebar untuk memilih peran
    with st.sidebar:
        st.image("logo_kopi.jpg", use_container_width=True)
        role = option_menu(
            "Pilih Peran:",
            ["Customer", "Admin"],
            icons=["person-circle", "person-badge"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "5px", "background-color": "#D1CFCF"},  # Light gray
                "icon": {"color": "#3E3E3E", "font-size": "20px"},  # Dark gray
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#A89F91"},  # Brown hover
                "nav-link-selected": {"background-color": "#6D435A"},  # Dark brown
            },
        )

    # Alur Customer (tanpa login)
    if role == "Customer":
        customer_menu = option_menu(
            "Menu Customer",
            ["Home", "Pick-Up Order", "Riwayat Pemesanan"],
            icons=["house", "bag-plus", "list-task"],
            menu_icon="menu-button",
            default_index=0,
            styles={
                "container": {"padding": "5px", "background-color": "#D1CFCF"},  # Light gray
                "icon": {"color": "#3E3E3E", "font-size": "18px"},  # Dark gray
                "nav-link": {"font-size": "14px", "text-align": "left", "margin": "0px", "--hover-color": "#A89F91"},  # Brown hover
                "nav-link-selected": {"background-color": "#6D435A"},  # Dark brown
            },
        )

        if customer_menu == "Home":
            home_page()
        elif customer_menu == "Pick-Up Order":
            pick_up_order_page()
        elif customer_menu == "Riwayat Pemesanan":
            order_history_page()

    # Alur Admin (dengan login)
    elif role == "Admin":
        if not st.session_state["logged_in"]:
            st.sidebar.subheader("Login Required")
            login_page()
        else:
            st.sidebar.success("You are logged in.")
            admin_menu = option_menu(
                "Menu Admin",
                ["Transaksi Data Coffee", "Model Algoritma", "Menu Kalori", "Logout"],
                icons=["cup-straw", "gear", "calculator", "box-arrow-right"],
                menu_icon="gear",
                default_index=0,
                styles={
                    "container": {"padding": "5px", "background-color": "#D1CFCF"},  # Light gray
                    "icon": {"color": "#3E3E3E", "font-size": "18px"},  # Dark gray
                    "nav-link": {"font-size": "14px", "text-align": "left", "margin": "0px", "--hover-color": "#A89F91"},  # Brown hover
                    "nav-link-selected": {"background-color": "#6D435A"},  # Dark brown
                },
            )

            if admin_menu == "Transaksi Data Coffee":
                coffee_menu_page()
            elif admin_menu == "Model Algoritma":
                model_algorithm_page()
            elif admin_menu == "Menu Kalori":
                menu_kalori_page()
            elif admin_menu == "Logout":
                st.session_state["logged_in"] = False
                st.sidebar.success("You have logged out.")

if __name__ == "__main__":
    main()
