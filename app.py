import streamlit as st
import yfinance as yf
import pandas as pd
import seaborn as sns
import json
import numpy as np
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from matplotlib import pyplot as plt
from util import transform_data

# Update daily via Yahoo API
voo = yf.Ticker("VOO")
data = voo.history(period='10Y')
# retrain on daily data
model = transform_data(data)

# Set title
st.title('Forecast VOO stock WebAPP📈')
st.write('Updated Daily via YahooAPI')


option = st.slider('How many days into future?', 100, 1000, 50)

if st.button('Forecast'):
    future = model.make_future_dataframe(periods=option, freq='D') # daily forecast
    forecast = model.predict(future)
    plot1 = model.plot(forecast)
    st.write(plot1)
    st.write(forecast[['ds', 'yhat_lower', 'yhat', 'yhat_upper']])
    st.write('Interpret: Model is confident that the open price of VOO will lie between yhat upper and yhat lower.')
