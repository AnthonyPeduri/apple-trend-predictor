import numpy as np
import pandas as pd
import tensorflow as tf
from Scraper import Scraper
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the saved model
model = tf.keras.models.load_model('apple_stock_model.h5')

apple_stock = Scraper()
#This will pull live data up to today
apple_history = apple_stock.get_all()

#This will pull data we used to get the predictions for results 4.
#apple_history = apple_stock.get_range("2003-10-30","2023-10-30")

# Preprocess the input data in the same way as done before training
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(apple_history[['Close']])

# Function to create the sequential data for prediction
def create_dataset(dataset, time_step=1):
    dataX = []
    for i in range(len(dataset) - time_step):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
    return np.array(dataX)

# Prepare the last n_input days as input for the prediction
time_step = 10  # same as 'n_input' used during training
last_days = scaled_data[-time_step:]
last_days = last_days.reshape(1, time_step, 1)

# Predict the next 30 days
future_predictions = []

for _ in range(30):
        # Make prediction using the last n_input days
    prediction = model.predict(last_days)
    
    # Reshape the prediction to match the input shape for the model
    prediction_reshaped = prediction.reshape(1, 1, 1)
    
    # Append the prediction as the next day's closing price
    future_predictions.append(prediction[0][0])
    
    # Add the prediction to the input sequence and remove the first day to keep the sequence length constant
    last_days = np.append(last_days[:, 1:, :], prediction_reshaped, axis=1)

# Inverse transform the predictions to the original price scale
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Generate dates for the predicted values
last_date = pd.to_datetime(apple_history.index[-1])
predicted_dates = pd.date_range(start=last_date, periods=30, freq='B')  # B for business days

# Create a DataFrame for the predicted stock prices
predicted_stock_prices = pd.DataFrame(future_predictions, index=predicted_dates, columns=['Predicted_Close'])

#Plot predictions
plt.figure(figsize=(10,5))
plt.plot(predicted_stock_prices.index, predicted_stock_prices['Predicted_Close'], marker='o', color='orange')
plt.title('30 Day Future Predictions of Apple Stock Closing Prices')
plt.xlabel('Date')
plt.ylabel('Predicted Closing Price (USD)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()  
plt.show()

