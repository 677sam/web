import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model_utils import prepare_data, load_model_and_predict

# Configuration for file path and feature columns
file_path = 'D:/download_browser/project/output.xlsx'
features = ['USD_CNY', 'Change_Rate', 'UST_10Y_Yield', 'China_10Y_Yield', 'Yield_Spread',
            'VIX_Index', 'SHIBOR_Rate', 'WTI_Crude_Oil', 'Dow_Jones_Index',
            'Gold_Futures', 'USD_Index', 'MSCI_Index']

# Page title and header
st.title("USD/CNY Exchange Rate Prediction")
st.write("This web app uses high-frequency data to predict the USD/CNY exchange rate for the next 7 days using advanced ARMA-GARCH and LSTM models.")

# Data processing and model loading
st.sidebar.header("Data & Model")
st.sidebar.write("Preparing data and loading the model...")
data, scaler = prepare_data(file_path, features)

# Prediction generation section
st.header("Generate Predictions")
if st.button("Generate Predictions"):
    st.write("Predicting future exchange rates...")
    predictions = load_model_and_predict(data, scaler, features)

    # Display predicted results
    st.subheader("Predicted USD/CNY Exchange Rates for the Next 7 Days:")
    for i, rate in enumerate(predictions, start=1):
        st.write(f"Day {i}: {rate:.4f}")

    # Visualization of predicted exchange rates
    st.write("Visualization of Predicted Exchange Rates")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, 8), predictions, marker='o', label="Predicted Rates")
    ax.set_title("Predicted USD/CNY Exchange Rates for the Next 7 Days")
    ax.set_xlabel("Days")
    ax.set_ylabel("Exchange Rate")
    ax.legend()
    st.pyplot(fig)

# Static image section for actual vs predicted comparison
st.header("Actual vs Predicted Comparison")
st.image('D:/download_browser/project/1733983428777.jpg', caption='LSTM: Actual vs Predicted USD/CNY Exchange Rate', use_column_width=True)

# Download predictions as CSV
st.header("Download Prediction Results")
sample_data = pd.DataFrame({
    'Day': [1, 2, 3, 4, 5, 6, 7],
    'Predicted USD/CNY': predictions if 'predictions' in locals() else [None] * 7
})
csv = sample_data.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Predictions as CSV",
    data=csv,
    file_name='predicted_exchange_rates.csv',
    mime='text/csv',
)

# Footer with additional notes
st.sidebar.markdown("### Note")
st.sidebar.write("The above chart demonstrates historical comparisons using ARMA-GARCH and LSTM models.")
st.sidebar.write("For more details, please refer to the documentation.")
