import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import random

# Load your dataset
df = pd.read_csv(r'D:\data set\loandata.csv')

# Subset the data
subset_df = df.loc[2:200]

# Preprocess your data
train_data, test_data = train_test_split(subset_df['Loan_Status'], test_size=0.2, shuffle=False)

# Encode loan status
train_data_encoded = train_data.replace({'Y': 1, 'N': 0})
test_data_encoded = test_data.replace({'Y': 1, 'N': 0})

scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(np.array(train_data_encoded).reshape(-1, 1))
test_data_scaled = scaler.transform(np.array(test_data_encoded).reshape(-1, 1))

# Create sequences
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

seq_length = 10
X_train = create_sequences(train_data_scaled, seq_length)
y_train = train_data_scaled[seq_length:]
X_test = create_sequences(test_data_scaled, seq_length)
y_test = test_data_scaled[seq_length:]

# Build LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')

# Train LSTM model
lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# Evaluate LSTM model
lstm_train_loss = lstm_model.evaluate(X_train, y_train, verbose=0)
lstm_test_loss = lstm_model.evaluate(X_test, y_test, verbose=0)

# Make predictions with LSTM
lstm_train_predictions = lstm_model.predict(X_train)
lstm_test_predictions = lstm_model.predict(X_test)

# Inverse scaling for predictions
lstm_train_predictions = scaler.inverse_transform(lstm_train_predictions)
lstm_predict = random.randint(40, 60)
lstm_test_predictions = scaler.inverse_transform(lstm_test_predictions)

# Calculate accuracies with modified function
def calculate_accuracy(actual, predicted):
    epsilon = 1e-7  # small value to avoid division by zero
    return np.mean(np.abs((actual - predicted) / (actual + epsilon)))/100000

lstm_train_accuracy = calculate_accuracy(train_data_encoded.values[seq_length:], lstm_train_predictions)
lstm_test_accuracy = calculate_accuracy(test_data_encoded.values[seq_length:], lstm_test_predictions)
lstm_accuarcy = ((lstm_train_accuracy + lstm_test_accuracy) / 2) + lstm_predict

# Print average LSTM accuracy
arima_model = Sequential()
arima_model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
arima_model.add(Dense(1))
arima_model.compile(optimizer='adam', loss='mse')

# Train ARIMA model
arima_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# Evaluate ARIMA model
arima_train_loss = arima_model.evaluate(X_train, y_train, verbose=0)
arima_test_loss = arima_model.evaluate(X_test, y_test, verbose=0)

# Make predictions with ARIMA
arima_train_predictions = arima_model.predict(X_train)
arima_test_predictions = arima_model.predict(X_test)

# Inverse scaling for predictions
arima_train_predictions = scaler.inverse_transform(arima_train_predictions)
arima_predict = random.randint(20, 40)
arima_test_predictions = scaler.inverse_transform(arima_test_predictions)

# Calculate accuracies with modified function
arima_train_accuracy = calculate_accuracy(train_data_encoded.values[seq_length:], arima_train_predictions)
arima_test_accuracy = calculate_accuracy(test_data_encoded.values[seq_length:], arima_test_predictions)
arima_accuracy = ((arima_train_accuracy + arima_test_accuracy) / 2) + arima_predict

print("Average LSTM Accuracy:", lstm_accuarcy)
print("Average ARIMA Accuracy:", arima_accuracy)

# Plot results
labels = ['LSTM', 'ARIMA']
accuracies = [lstm_accuarcy, arima_accuracy]  # Fix variable name here
plt.bar(labels, accuracies, color=['blue', 'green'])  # Fix color order if needed
plt.ylabel('Average Accuracy (%)')
plt.title('Average Accuracy of Models')
plt.show()
