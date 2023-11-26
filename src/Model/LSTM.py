import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load and preprocess your data
file_path = '/Users/anthonypeduri/Development/Apple-Trend-Predictor/apple-trend-predictor/src/Data/apple-data.csv'
apple_stock_data = pd.read_csv(file_path)
apple_stock_data['Date'] = pd.to_datetime(apple_stock_data['Date'])
apple_stock_data.set_index('Date', inplace=True)

# Define the training and testing period
train_end_date = pd.to_datetime('2023-06-13')
test_start_date = pd.to_datetime('2023-06-14')
test_end_date = pd.to_datetime('2023-07-13')

train_data = apple_stock_data[apple_stock_data.index <= train_end_date]
test_data = apple_stock_data[(apple_stock_data.index >= test_start_date) & (apple_stock_data.index <= test_end_date)]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data[['AAPL']])
scaled_test_data = scaler.transform(test_data[['AAPL']])

# Function to create the dataset for LSTM
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# Prepare data for LSTM
look_back = 1
trainX, trainY = create_dataset(scaled_train_data, look_back)
testX, testY = create_dataset(scaled_test_data, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Build and train the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

early_stop = EarlyStopping(monitor='val_loss', patience=5)
model.fit(trainX, trainY, epochs=5, batch_size=5, verbose=2, callbacks=[early_stop], validation_data=(testX, testY))

# Predict on the test data
test_predictions = model.predict(testX)
test_predictions = scaler.inverse_transform(test_predictions)

# Calculate RMSE for accuracy
rmse = sqrt(mean_squared_error(test_data[look_back:]['AAPL'], test_predictions))
print(f'Root Mean Squared Error: {rmse}')

# Calculate directional accuracy
direction_correct = np.sum(np.sign(test_predictions[1:] - test_predictions[:-1]) == np.sign(test_data[look_back:]['AAPL'].values[1:] - test_data[look_back:]['AAPL'].values[:-1]))
directional_accuracy = direction_correct / (len(test_predictions) - 1)
print(f'Directional Accuracy: {directional_accuracy * 100:.2f}%')

# Calculate normal accuracy
actual_prices = test_data['AAPL'].values[-len(test_predictions):]
normal_accuracy = np.mean(np.abs((test_predictions - actual_prices) / actual_prices))
print(f'Normal Accuracy: {normal_accuracy * 100:.2f}%')

# Create forecast DataFrame
forecast_dates = pd.date_range(start=test_start_date, periods=len(test_predictions), freq='B')
forecast_df = pd.DataFrame(data=test_predictions.flatten(), index=forecast_dates, columns=['Predicted Closing Price'])

# Plotting only the last month's actual and forecasted closing prices
plt.figure(figsize=(10, 5))
plt.plot(test_data['AAPL'], label='Actual Closing Prices', color='green')
plt.plot(forecast_df, label='Forecasted Closing Prices', color='orange')
plt.title('Actual vs Forecasted Apple Stock Prices (June 13th to July 13th, 2023)')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()

# Plotting only the last month's actual and forecasted closing prices
plt.figure(figsize=(10, 5))
plt.plot(test_data['AAPL'], label='Actual Closing Prices', color='green')
plt.plot(forecast_df, label='Forecasted Closing Prices', color='orange')
plt.title('Actual vs Forecasted Apple Stock Prices (June 13th to July 13th, 2023)')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()