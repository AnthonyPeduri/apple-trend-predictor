# Re-importing necessary libraries and loading the data
import numpy as np
import pandas as pd
import tensorflow as tf
from Scraper import Scraper
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


apple_stock = Scraper()
apple_history = apple_stock.get_all()
print(apple_history)

apple_history["AAPL_10"] = apple_history['Close'].rolling(10).mean()
apple_history["AAPL_30"] = apple_history['Close'].rolling(30).mean()
apple_history["AAPL_50"] = apple_history['Close'].rolling(50).mean()

print(apple_history)
apple = apple_history[["Close", "AAPL_10", "AAPL_30", "AAPL_50"]]
split_date = '2020-10-30'
apple_train = apple.loc[apple.index <= split_date].copy()
apple_test = apple.loc[apple.index > split_date].copy()

# Normalize the Close prices for training and testing data
scaler = MinMaxScaler(feature_range=(0, 1))
apple_train_scaled = scaler.fit_transform(apple_train[['Close']])
apple_test_scaled = scaler.transform(apple_test[['Close']])

# Define the number of inputs for the LSTM model
n_input = 10

# Function to prepare training data for LSTM
def Sequential_Input_LSTM(df_scaled, input_sequence):
    X, y = [], []
    for i in range(len(df_scaled) - input_sequence):
        X.append(df_scaled[i:(i + input_sequence), 0])
        y.append(df_scaled[i + input_sequence, 0])
    return np.array(X), np.array(y)

X_train, y_train = Sequential_Input_LSTM(apple_train_scaled, n_input)
X_test, y_test = Sequential_Input_LSTM(apple_test_scaled, n_input)

# Function to predict a single future day
def predict_next_day(last_sequence, model, scaler):
    last_sequence = last_sequence.reshape((1, n_input, 1))
    next_day_price = model.predict(last_sequence)
    return next_day_price[0][0]

# Load the pre-trained LSTM model
# The model is expected to be saved and loaded here. For the sake of demonstration, we will assume the model is named 'apple_stock_model.h5'.
model = tf.keras.models.load_model("src/Model/apple_stock_model.h5")

# Predict 30 days into the future
new_predictions = []
current_sequence = apple_test_scaled[-n_input:].flatten()  # Get the last known sequence of stock prices

for _ in range(30):
    next_day_price = predict_next_day(current_sequence, model, scaler)
    new_predictions.append(next_day_price)
    current_sequence = np.append(current_sequence[1:], next_day_price)  # Update the sequence

# Rescale the predictions back to the original price range
new_predictions_rescaled = scaler.inverse_transform(np.array(new_predictions).reshape(-1, 1)).flatten()

# Create a date range for the next 30 days
last_date = apple_test.index[-1]  # Assuming this is a pandas Timestamp object
date_range_future = pd.date_range(start=last_date, periods=31)[1:]

# Assuming n_input is the number of input days your model uses to make a prediction
# You need to slice the apple_test.index to start from the same point as your predictions

# After model prediction
if len(X_test.shape) == 2:
    X_test = X_test.reshape((X_test.shape[0], n_input, 1))


y_pred = model.predict(X_test)

# Rescale the predictions back to the original prices
y_pred_rescaled = scaler.inverse_transform(y_pred)
# Plotting actual, forecasted, and future predicted closing prices
plt.figure(figsize=(10, 5))
# Adjusting the apple_test.index to match the shape of the y data
plt.plot(apple_test.index[n_input:], scaler.inverse_transform(apple_test_scaled[n_input:]), label='Actual Closing Prices', color='green')
plt.plot(apple_test.index[n_input:], y_pred_rescaled, label='Forecasted Closing Prices', color='orange')
plt.plot(date_range_future, new_predictions_rescaled, label='Future Predicted Prices', color='blue')
plt.title('Actual vs Forecasted vs Future Predicted Apple Stock Prices')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
