import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from keras.models import load_model

st.set_page_config(page_title="Dogecoin Price Predictor", layout="wide")

st.title("🐶📈 Dogecoin Price Prediction with LSTM")

# Load model and scaler
@st.cache_resource
def load_artifacts():
    model = load_model("lstm_stock_model.keras")
    scaler = joblib.load("scaler.save")
    return model, scaler

model, scaler = load_artifacts()

# Upload CSV
uploaded_file = st.file_uploader("📤 Upload Dogecoin CSV (must contain 'Date' and 'Close')", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    try:
        df['Date'] = pd.to_datetime(df['Date'])
    except:
        st.error("❌ 'Date' column could not be parsed.")
        st.stop()

    df.sort_values("Date", inplace=True)

    st.subheader("📊 Raw Data Preview")
    st.dataframe(df.tail())

    close_prices = df['Close'].values.reshape(-1, 1)
    scaled_data = scaler.transform(close_prices)

    sequence_length = 60
    X_test = []
    for i in range(sequence_length, len(scaled_data)):
        X_test.append(scaled_data[i-sequence_length:i, 0])

    X_test = np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Predictions
    predictions = model.predict(X_test)

    # Inverse transform predictions
    predictions_extended = np.concatenate((predictions, np.zeros_like(predictions)), axis=1)
    predicted_prices = scaler.inverse_transform(predictions_extended)[:, 0]

    actual_prices = df['Close'].values[sequence_length:]

    st.subheader("📈 Actual vs Predicted Prices")
    fig1 = plt.figure(figsize=(12, 6))
    plt.plot(df['Date'].values[sequence_length:], actual_prices, label='Actual')
    plt.plot(df['Date'].values[sequence_length:], predicted_prices, label='Predicted')
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.title("Dogecoin Price Prediction")
    plt.legend()
    st.pyplot(fig1)

    # Forecasting future prices
    st.subheader("🔮 Forecast Future Prices")
    future_days = st.slider("How many days to predict?", min_value=1, max_value=30, value=7)

    last_sequence = scaled_data[-sequence_length:]
    current_sequence = last_sequence.reshape(1, sequence_length, 1)

    future_preds = []
    for _ in range(future_days):
        pred = model.predict(current_sequence)[0][0]
        future_preds.append(pred)

        # Update the sequence with new prediction
        current_sequence = np.append(current_sequence[:, 1:, :], [[[pred]]], axis=1)

    # Inverse transform future predictions
    future_preds_np = np.array(future_preds).reshape(-1, 1)
    future_preds_ext = np.concatenate((future_preds_np, np.zeros_like(future_preds_np)), axis=1)
    future_prices = scaler.inverse_transform(future_preds_ext)[:, 0]

    # Generate future dates
    last_date = df['Date'].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_days)

    fig2 = plt.figure(figsize=(10, 4))
    plt.plot(future_dates, future_prices, marker='o', color='tomato', label="Forecast")
    plt.title("Future Dogecoin Prices")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True)
    plt.legend()
    st.pyplot(fig2)

    st.success("✅ Prediction complete!")

else:
    st.info("📎 Upload a valid Dogecoin CSV file to begin.")
