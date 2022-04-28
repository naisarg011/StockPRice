# THIS FILE IS USED TO TRAIN MODEL


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as data
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
# from keras.models import load_model
# import streamlit as st
from sklearn.preprocessing import MinMaxScaler

start = '2010-01-01'
end = '2021-12-31'

df = data.DataReader('^BSESN', 'yahoo', start, end)
df = df.reset_index()
df = df.drop(['Date', 'Adj Close'], axis=1)
# print(df.head())

ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
# plt.figure(figsize=(12, 6))
# plt.plot(df.Close)
# plt.plot(ma100, 'r')
# plt.plot(ma200, 'g')
# plt.show()
# print(df.shape) to show the total value of total output data

# Training and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])
# print(data_testing.shape)
# print(data_training.shape)

scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)
# print(data_training_array)

x_train = []
y_train = []

for i in range(100, data_training.shape[0]):
    x_train.append(data_training_array[i - 100: i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# print(x_train)


# ML MODEL LOGIC


model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model = Sequential()
model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))

model = Sequential()
model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))

model = Sequential()
model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=75)
model.build()
model.save('stock_model2.h5')

past_100_days = data_training.tail(100)

final_df = past_100_days.append(data_testing, ignore_index=True)

input_data = scaler.transform(final_df)

# print(input_data.shape)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# predictions
print(x_test.shape)
y_prediction = model.predict(x_test)
# print(scaler.scale_)
print(y_prediction.shape)

# y_prediction = y_prediction / scaler.scale_
# y_test = y_test / scaler.scale_
#
# print(y_test)
#
# plt.figure(figsize=(12, 6))
# plt.plot(y_test, 'b', label='Original Price')
# plt.plot(y_prediction, 'r', label='Predicted Value')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend()
# plt.show()
