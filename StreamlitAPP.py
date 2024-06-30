import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
data_path = "D:/Projektiki/predykcja_rezerwacji/hotel_booking.csv"
df = pd.read_csv(data_path)

# Загрузка моделей
lr_model = joblib.load("D:/Projektiki/predykcja_rezerwacji/logistic_regression_model.pkl")
xgb_model = joblib.load("D:/Projektiki/predykcja_rezerwacji/xgboost_model.pkl")
tf_model = tf.keras.models.load_model("D:/Projektiki/predykcja_rezerwacji/tf_model.h5")


# Функция для предсказания
def make_predictions(input_data):
    # Здесь может быть код для масштабирования данных (scaler)
    # X_scaled = scaler.transform(input_data)

    y_pred_lr = lr_model.predict(input_data)
    y_pred_xgb = xgb_model.predict(input_data)
    y_pred_tf = (tf_model.predict(input_data) > 0.5).astype(int)

    return y_pred_lr, y_pred_xgb, y_pred_tf


# Интерфейс Streamlit
st.title("Анализ и Предсказание Отмены Бронирования Отелей")

st.sidebar.header("Настройки")
section = st.sidebar.selectbox("Выберите раздел", ["EDA", "Предсказание"])

if section == "EDA":
    st.header("Анализ данных")
    variable = st.selectbox("Выберите переменную для анализа", df.columns)
    plot_type = st.selectbox("Тип графика", ["Гистограмма", "Boxplot", "Scatterplot"])

    if plot_type == "Гистограмма":
        plt.figure(figsize=(10, 6))
        sns.histplot(df[variable], bins=30)
        st.pyplot(plt)
    elif plot_type == "Boxplot":
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='is_canceled', y=variable, data=df)
        st.pyplot(plt)
    elif plot_type == "Scatterplot":
        scatter_var = st.selectbox("Выберите вторую переменную для scatterplot", df.columns)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[variable], y=df[scatter_var])
        st.pyplot(plt)

elif section == "Предсказание":
    st.header("Предсказание отмены бронирования")
    lead_time = st.number_input("Lead Time")
    adr = st.number_input("ADR")
    total_of_special_requests = st.number_input("Total of Special Requests")

    input_data = pd.DataFrame([[lead_time, adr, total_of_special_requests]],
                              columns=["lead_time", "adr", "total_of_special_requests"])

    if st.button("Предсказать"):
        y_pred_lr, y_pred_xgb, y_pred_tf = make_predictions(input_data)

        st.write("Предсказания:")
        st.write(f"Логистическая регрессия: {y_pred_lr[0]}")
        st.write(f"XGBoost: {y_pred_xgb[0]}")
        st.write(f"Нейронная сеть: {y_pred_tf[0][0]}")



