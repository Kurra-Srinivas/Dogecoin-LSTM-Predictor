# 🐶📈 Dogecoin Price Predictor (LSTM-based)

A deep learning project using LSTM (Long Short-Term Memory) networks to forecast Dogecoin (DOGE) prices based on historical data. This project leverages the power of sequence modeling and time series forecasting to analyze crypto market trends.

---

## 🚀 Project Overview

- 📊 Input Data: Historical DOGE-USD prices
- 🤖 Model Type: LSTM Neural Network (with Keras & TensorFlow backend)
- 🧠 Features: Closing price series (scaled), time-based sequence modeling
- 🎯 Objective: Predict future closing prices of Dogecoin
- 📉 Visualizations: Loss curve, actual vs predicted, residual diagnostics, future price projection

---

## 📌 Highlights

- ✅ Built and trained a custom LSTM network on preprocessed crypto time series
- ✅ Achieved strong performance on test set (R² score, low residuals)
- ✅ Visualized residual distribution, model diagnostics, and future forecasts
- ✅ Saved model and scaler for real-time or batch inference
- ✅ Ready for Streamlit deployment!

---

## 🧰 Tech Stack

| Tool/Library     | Purpose                         |
|------------------|---------------------------------|
| Python (3.12)    | Core programming language       |
| TensorFlow/Keras | LSTM Model Building & Training  |
| Scikit-learn     | Metrics, preprocessing          |
| Pandas, NumPy    | Data manipulation               |
| Matplotlib, Seaborn | Plotting & visualization    |
| Git/GitHub       | Version control & collaboration |

---

## 📂 Project Structure

```
📁 Dogecoin-LSTM-Predictor/
│
├── DOGE-USD.csv                   # Raw historical Dogecoin data
├── dogecoin_lstm.ipynb           # Main Jupyter Notebook (core logic)
├── lstm_stock_model.keras        # Trained LSTM model (Keras format)
├── scaler.save                   # Saved MinMaxScaler for future inference
│
├── actual_vs_predicted.png       # Visualization of model performance
├── residual_distribution.png     # Histogram + KDE of residuals
├── residual_diagnostics.png      # Residuals vs predicted plot
├── loss_curve.png                # Training loss progression
├── future_price_prediction.png   # Future forecasted prices
│
└── README.md                     # This file ✨
```

---

## 📈 Sample Results

| Metric            | Value     |
|-------------------|-----------|
| R² Score (Test)   | `~0.90+` (adjust as per actual) |
| MAE / MSE         | Low (fine-tune in model tuning) |

---

## 🧪 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/Dogecoin-LSTM-Predictor.git
cd Dogecoin-LSTM-Predictor

# 2. (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install requirements
pip install -r requirements.txt

# 4. Run the notebook
jupyter notebook dogecoin_lstm.ipynb
```

---

## 🌍 Deployment Ideas

✅ Streamlit App  
✅ Flask or FastAPI backend with a frontend dashboard  
✅ Scheduled inference jobs using `cron` or Airflow

---

## 🙋‍♂️ Author

**Kurra Srinivas**  
📧 srinivaskurra886@gmail.com  
📎 [LinkedIn](https://www.linkedin.com) *(replace with actual)*  
🐙 GitHub: [Kurra-Srinivas](https://github.com/Kurra-Srinivas)

---

## ⭐ Star This Repo

If you found this project helpful or inspiring, give it a ⭐️ to show support!