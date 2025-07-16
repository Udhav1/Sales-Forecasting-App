import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.title("📈 Sales Forecasting with Prophet")

uploaded_file = st.file_uploader("Upload your train.csv file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Prepare data
    df['date'] = pd.to_datetime(df['date'])
    daily_sales = df.groupby('date').agg({'sales': 'sum'}).reset_index()
    prophet_df = daily_sales.rename(columns={'date': 'ds', 'sales': 'y'})
    
    # Forecast horizon input
    periods_input = st.number_input('How many days to forecast?', min_value=30, max_value=365, value=180)

    # Train model
    model = Prophet()
    model.fit(prophet_df)

    # Create future dataframe
    future = model.make_future_dataframe(periods=periods_input)
    forecast = model.predict(future)

    # Plot forecast
    st.subheader("Forecast Plot")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    # Plot components
    st.subheader("Forecast Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

    # Show forecasted data
    st.subheader("Forecasted Data")
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())