import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

data = pd.read_csv('all_stocks_5yr.csv')

# Selecting first 5 unique companies
companies = data['Name'].unique()[:5]

# Create a sample of the data for the selected companies
sample_data = data[data['Name'].isin(companies)].copy()

# Convert 'date' column to datetime
sample_data['date'] = pd.to_datetime(sample_data['date'])

# Plotting stock prices over time for selected companies
plt.figure(figsize=(18, 12))
plt.suptitle('Stock Prices Over Time', fontsize=16)
plt.tight_layout(pad=3.0)

for index, company in enumerate(companies, 1):
    plt.subplot(3, 3, index)
    c = sample_data[sample_data['Name'] == company]
    plt.plot(c['date'], c['close'], c="r", label="Close", marker="o", linestyle="-")
    plt.plot(c['date'], c['open'], c="g", label="Open", marker="^", linestyle="--")
    plt.title(company)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

plt.show()

# Plotting volume over time for selected companies
plt.figure(figsize=(18, 12))
plt.suptitle('Trading Volume Over Time', fontsize=16)
plt.tight_layout(pad=3.0)

for index, company in enumerate(companies, 1):
    plt.subplot(3, 3, index)
    c = sample_data[sample_data['Name'] == company]
    plt.plot(c['date'], c['volume'], c='purple', marker='*')
    plt.title(f"{company} Volume")
    plt.xlabel('Date')
    plt.ylabel('Volume')

plt.show()

# Plotting Apple stock prices within a specific date range
apple = data[data['Name'] == 'AAPL'].copy()
apple['date'] = pd.to_datetime(apple['date']) 

apple_filtered = apple[(apple['date'] > datetime(2013, 1, 1)) & (apple['date'] < datetime(2018, 1, 1))]

plt.figure(figsize=(12, 8))
plt.plot(apple_filtered['date'], apple_filtered['close'], marker='o', linestyle='-', color='b')
plt.title('Apple Stock Prices (2013-2018)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.grid(True)
plt.show()

# Prepare data for LSTM model
close_data = apple.filter(['close'])
dataset = close_data.values
training = int(np.ceil(len(dataset) * .95))
print(training)

# Scale 
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:int(training), :]

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Define LSTM model
model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=64))
model.add(Dense(32))
model.add(Dropout(0.5))
model.add(Dense(1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error') 
history = model.fit(x_train, y_train, epochs=10)

test_data = scaled_data[training - 60:, :] 
x_test = [] 
y_test = dataset[training:, :] 
for i in range(60, len(test_data)): 
    x_test.append(test_data[i-60:i, 0]) 

x_test = np.array(x_test) 
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) 

#predict the testing data 
predictions = model.predict(x_test) 
predictions = scaler.inverse_transform(predictions) 

# evaluation  
mse = np.mean(((predictions - y_test) ** 2)) 
print("MSE", mse) 
print("RMSE", np.sqrt(mse)) 

train = apple[:training].copy()
test = apple[training:].copy()
test.loc[:, 'Predictions'] = predictions 

# ploting
plt.figure(figsize=(10, 8)) 
plt.plot(train['date'], train['close'], label='Train') 
plt.plot(test['date'], test['close'], label='Test') 
plt.plot(test['date'], test['Predictions'], label='Predictions') 
plt.title('Apple Stock Close Price') 
plt.xlabel('Date') 
plt.ylabel("Close") 
plt.legend()
plt.show()
