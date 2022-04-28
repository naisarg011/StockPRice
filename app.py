# THIS FILE IS USE THE TRAINED MODEL AND PREDICT THE PRICE


import numpy as np
import pandas as pd
import pandas_datareader as data
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

start = '2015-01-01'
end = '2021-12-31'

st.title('Asset Trend Prediction')

user_input = st.text_input('Enter Asset Ticker', 'BTC-USD')
df = data.DataReader(user_input, 'yahoo', start, end)

st.subheader('Data From 2015-2021')
st.write(df.describe())

st.subheader('Closing Price ---> Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price ---> Time Chart With 100MA')
MA100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(MA100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price ---> Time Chart With 100MA & 200MA')
MA200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(MA100, 'r')
plt.plot(MA200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])

scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# ML MODEL LOADING

model = load_model('stock_model2.h5')

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_pridicted = model.predict(x_test)

scaler = scaler.scale_

scaler_factor = 1 / scaler[0]
y_pridicted = y_pridicted * scaler_factor
y_test = y_test * scaler_factor


st.subheader('Prediction ---> original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_pridicted, 'r', label='Predicted Value')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

