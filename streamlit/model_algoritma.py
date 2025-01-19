import streamlit as st
import pandas as pd
import mysql.connector
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np


def create_connection():
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="coffee_db"
    )
    return connection


def model_algorithm_page():
    st.subheader("Latih Model PCA dan SVM untuk Prediksi Kalori")

    # Menghubungkan ke database dan mengambil data kopi
    connection = create_connection()
    coffee_data = pd.read_sql("SELECT * FROM menu_coffee", connection)
    connection.close()

    # Menampilkan DataFrame dengan data menu coffee
    st.subheader("Data Menu Coffee")
    if not coffee_data.empty:
        st.dataframe(coffee_data)

        # Proses data dan latih model
        pca, svm, accuracy, precision, recall, conf_matrix, predictions, coffee_data_clean = train_model(coffee_data)

        # Menambahkan tombol untuk menampilkan tabel hasil prediksi
        if st.button("Tampilkan Hasil Prediksi"):
            st.subheader("Hasil Prediksi untuk Semua Item Kopi")

            # Tentukan threshold kalori untuk klasifikasi
            calorie_threshold = coffee_data_clean['calories'].median()


            # Klasifikasikan hasil prediksi
            labels = ['Kalori Rendah' if p < calorie_threshold else 'Kalori Tinggi' for p in predictions]
            true_labels = ['Kalori Rendah' if c < calorie_threshold else 'Kalori Tinggi' for c in coffee_data_clean['calories']]

            # Buat DataFrame dengan label
            prediction_df = pd.DataFrame({
                'Item': coffee_data_clean['item'],
                'Prediksi Kalori': predictions,
                'Kategori': labels,
                'Prediksi Label': true_labels
            })

            # Tampilkan hasil prediksi dalam bentuk dataframe
            st.write(prediction_df)

            # Visualisasi PCA hanya untuk prediksi SVM (kalori tinggi/rendah)
            st.subheader("Visualisasi PCA Berdasarkan Prediksi Kalori (SVM) dengan Hyperplane")
            plot_svm_pca(coffee_data_clean, pca, predictions, calorie_threshold, svm)

            # Evaluasi model klasifikasi
            st.subheader("Evaluasi Model")
            st.write(f"Accuracy: {accuracy:.2f}")
            st.write(f"Precision: {precision:.2f}")
            st.write(f"Recall: {recall:.2f}")

           # Visualisasi Confusion Matrix
            st.subheader("Confusion Matrix")
            ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Kalori Rendah', 'Kalori Tinggi']).plot(cmap='Blues')
            st.pyplot(plt)

    else:
        st.warning("Tidak ada data yang tersedia untuk pelatihan.")


def train_model(data):
    data_clean = data.dropna(subset=['calories'])
    X = data_clean.drop(columns=['item', 'tanggal_transaksi', 'customer'])
    y = data_clean['calories']


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.5],
        'kernel': ['linear', 'rbf']
    }

    svm = SVR()
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    best_svm = grid_search.best_estimator_
    y_pred = best_svm.predict(X_test)

    calorie_threshold = y.median()
    y_pred_labels = ['Kalori Rendah' if p < calorie_threshold else 'Kalori Tinggi' for p in y_pred]
    y_test_labels = ['Kalori Rendah' if c < calorie_threshold else 'Kalori Tinggi' for c in y_test]

    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    precision = precision_score(y_test_labels, y_pred_labels, pos_label='Kalori Tinggi', average='binary')
    recall = recall_score(y_test_labels, y_pred_labels, pos_label='Kalori Tinggi', average='binary')
    conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)

    # Prediksi untuk semua data
    predictions = best_svm.predict(X_pca)

    return pca, best_svm, accuracy, precision, recall, conf_matrix, predictions, data_clean


def plot_svm_pca(data, pca, predictions, calorie_threshold, svm_model):
    # Transformasi data menggunakan PCA
    pca_components = pca.transform(StandardScaler().fit_transform(data.drop(columns=['item', 'tanggal_transaksi', 'customer'])))

    # Tentukan label berdasarkan threshold kalori
    labels = ['Kalori Rendah' if p < calorie_threshold else 'Kalori Tinggi' for p in predictions]

    # Tentukan grid untuk visualisasi
    x_min, x_max = pca_components[:, 0].min() - 1, pca_components[:, 0].max() + 1
    y_min, y_max = pca_components[:, 1].min() - 1, pca_components[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Prediksi pada grid untuk menggambar hyperplane
    Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot hyperplane sebagai garis pemisah
    plt.contour(xx, yy, Z, levels=[calorie_threshold], colors='black', linestyles='--', linewidths=1.5, label='Hyperplane')

    # Plot data points
    sns.scatterplot(
        x=pca_components[:, 0],
        y=pca_components[:, 1],
        hue=labels,
        palette={'Kalori Rendah': 'blue', 'Kalori Tinggi': 'red'},
        edgecolor='w',  # Add white edges to the scatter points
        linewidth=0.5    # Point edge width
    )

    plt.title("PCA Komponen Berdasarkan Prediksi Kalori (SVM) dengan Hyperplane")
    plt.xlabel("Kalori Rendah")
    plt.ylabel("Kalori Tinggi")
    plt.legend(title="Kategori")
    st.pyplot(plt)

