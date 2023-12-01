import math
import pandas as pd
import numpy as np
import tensorflow as tf
from Scraper import Scraper
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

apple_stock = Scraper("aapl", "20y")
apple_history = apple_stock.getAll()
print(apple_history)
# Load and preprocess the data
# df = pd.read_csv("src/Data/sep100.csv")
# df["Date"] = pd.to_datetime(df.Date)
# df = df.set_index("Date")
#top_df = df[["MSFT", "AAPL", "GOOG", "ACN"]].copy()
apple_history["AAPL_10"] = apple_history['Close'].rolling(10).mean()
apple_history["AAPL_30"] = apple_history['Close'].rolling(30).mean()
apple_history["AAPL_50"] = apple_history['Close'].rolling(50).mean()

print(apple_history)
apple = apple_history[["Close", "AAPL_10", "AAPL_30", "AAPL_50"]]
split_date = '2020-01-01'
apple_train = apple.loc[apple.index <= split_date].copy()
apple_test = apple.loc[apple.index > split_date].copy()

# Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
apple_train_scaled = scaler.fit_transform(apple_train[['Close']])
apple_test_scaled = scaler.transform(apple_test[['Close']])  # Use the same scaler to ensure consistency

n_input = 10

# Preparing training data for LSTM
def Sequential_Input_LSTM(df_scaled, input_sequence):
    X, y = [], []
    for i in range(len(df_scaled) - input_sequence):
        X.append(df_scaled[i:(i + input_sequence), 0])
        y.append(df_scaled[i + input_sequence, 0])
    return np.array(X), np.array(y)

X_train, y_train = Sequential_Input_LSTM(apple_train_scaled, n_input)
X_test, y_test = Sequential_Input_LSTM(apple_test_scaled, n_input)

# Define the LSTM model
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer((n_input, 1)),
    tf.keras.layers.LSTM(300, activation="tanh", return_sequences=True),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(200, activation="tanh"),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(40, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam',
              loss='mean_absolute_error',
              metrics=['mean_absolute_error'])

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_data=(X_test, y_test),
    verbose=1,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
)
model.save('apple_stock_model.h5')
# Generate predictions
y_pred = model.predict(X_test)

# Rescale the predictions back to the original prices
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE and MAE
rmse = math.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
mae = np.mean(np.abs(y_test_rescaled - y_pred_rescaled))

# Print the metrics
print(f'Test RMSE: {rmse:.2f}')
print(f'Test MAE: {mae:.2f}')

# Rescale the predictions back to the original prices
y_pred_rescaled = scaler.inverse_transform(y_pred)

# Adjust the test data to have the correct date index
trimmed_test_index = apple_test.index[n_input:]

# Plotting actual and forecasted closing prices
plt.figure(figsize=(10, 5))
plt.plot(trimmed_test_index, scaler.inverse_transform(apple_test_scaled[n_input:]), label='Actual Closing Prices', color='green')
plt.plot(trimmed_test_index, y_pred_rescaled, label='Forecasted Closing Prices', color='orange')
plt.title('Actual vs Forecasted Apple Stock Prices')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
