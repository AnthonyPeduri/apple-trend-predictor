import math
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


df = pd.read_csv("src/Data/sep100.csv")
df["Date"] = pd.to_datetime(df.Date)
df = df.set_index("Date")
df.head()

top_df = df[["MSFT", "AAPL", "GOOG", "ACN"]].copy()
top_df["AAPL_10"]= top_df["AAPL"].rolling(10).mean()
top_df["AAPL_30"]= top_df["AAPL"].rolling(30).mean()
top_df["AAPL_50"]= top_df["AAPL"].rolling(50).mean()

apple = top_df[["AAPL", "AAPL_10", "AAPL_30", "AAPL_50"]]
split_date = '2021-01-01'
apple_train = apple.loc[apple.index <= split_date].copy()
apple_test = apple.loc[apple.index > split_date].copy()
split_date = pd.to_datetime(split_date)

split_date = '2021-01-01'
split_date = pd.to_datetime(split_date)

apple_train = apple.loc[apple.index <= split_date].copy()
apple_test = apple.loc[apple.index > split_date].copy()

plt.figure(figsize=(10, 5))
plt.plot(apple_test.index, apple_test['AAPL'], '.', markersize=1, label='Test Set')
plt.plot(apple_train.index, apple_train['AAPL'], '.', markersize=1, label='Training Set')

plt.axvline(x=split_date, color='black', linestyle='--', label='Split Date')

plt.title('Train - test split (90 - 10)')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

df_min_model_data = df[['AAPL']]  # This creates a new DataFrame with only 'AAPL' column
df_min_model_data.index = pd.to_datetime(df_min_model_data.index)  # Convert the index to datetime

def Sequential_Input_LSTM(df, input_sequence):
    df_np = df.to_numpy()
    X = []
    y = []
    
    for i in range(len(df_np) - input_sequence):
        row = [a for a in df_np[i:i + input_sequence]]
        X.append(row)
        label = df_np[i + input_sequence]
        y.append(label)
        
    return np.array(X), np.array(y)


n_input = 10      


X, y = Sequential_Input_LSTM(df_min_model_data, n_input)

# Training data
x_train, y_train = X[:4500], y[:4500]

# Validation data
x_test, y_test = X[4500:], y[4500:]

EPOCHS = 20
BATCH_SIZE = 15

n_features = 1

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer((n_input,n_features)))
model.add(tf.keras.layers.LSTM(220, activation = "tanh",return_sequences=True))
model.add(tf.keras.layers.LSTM(100, activation ="tanh",return_sequences=False))
model.add(tf.keras.layers.Dense(40, activation = "relu"))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer=tf.keras.optimizers.Adam
                      (
                        learning_rate=1e-04,
                        beta_1=0.92,
                        beta_2=0.999,
                        epsilon=1e-07,
                        amsgrad=False,
                        weight_decay=None,
                        clipnorm=None,
                        clipvalue=None,
                        global_clipnorm=None,
                        use_ema=False,
                        ema_momentum=0.99
                      ),  
              metrics =["mean_absolute_error"],
              loss=tf.keras.losses.MeanAbsoluteError()
             )

early_stopping = tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=4,
                        mode="auto",
                        restore_best_weights=False,
)

history = model.fit(x = x_train,
                    y = y_train,
                    epochs = EPOCHS,
                    batch_size = BATCH_SIZE,
                    callbacks = early_stopping,
                               )

# Generate predictions
y_pred = model.predict(x_test)

# Evaluate the model on the test data
test_loss, test_mae = model.evaluate(x_test, y_test)

# Calculate RMSE
rmse = math.sqrt(mean_squared_error(y_test, y_pred))

# Print the results
print(f"Test MAE: {test_mae}")
print(f"Test RMSE: {rmse}")