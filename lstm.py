# Library
import pandas as pd 
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
def create_dataset(dataset, look_back=1):
	"""
	convert the value to dataset matrix
	"""
	data_u, data_time = [], []
	for i in range(len(dataset)-look_back):
		z = dataset[i:(i+look_back), 0]
		data_u.append(z)
		data_time.append(dataset[i+look_back, 0])
	return np.array(data_u), np.array(data_time)

#data
dataframe = pd.read_csv('100000data.csv', usecols=[1], engine='python')
dataset = dataframe.values

#data need to be in the range of 0-1
normalize_data = MinMaxScaler(feature_range=(0, 1))
dataset = normalize_data.fit_transform(dataset)

#data need to be split into training data and test data
training_data = int(len(dataset) * 0.7)
train, test = dataset[0:training_data ], dataset[training_data:len(dataset),:]
print(len(train), len(test))

#make u=t and time_step=t+1
look_back = 1
train_u, train_time = create_dataset(train, look_back)
test_u, test_time = create_dataset(test, look_back)

#make data input as [x, features, time steps]
train_u = np.reshape(train_u, (train_u.shape[0], 1, train_u.shape[1]))
test_u = np.reshape(test_u, (test_u.shape[0], 1, test_u.shape[1]))

#create LSTM network using keras library
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_u, train_time, epochs=50, batch_size=1, verbose=2)

#predictions
prediction = model.predict(test_u)

#inverse the data to original value
prediction = normalize_data.inverse_transform(prediction)
test_time = normalize_data.inverse_transform([test_time])

# calculate root mean squared error
test_error_score = np.sqrt(mean_squared_error(test_time[0], prediction[:,0]))
print('Test Error Score: %.2f RMSE' % (test_error_score))

# shift test predictions for plotting
prediction_plot = np.empty_like(dataset)
prediction_plot[:, :] = np.nan
prediction_plot[len(train_u)+(look_back):len(dataset)-1, :] = prediction

# plot baseline and predictions
plt.plot(normalize_data.inverse_transform(dataset))
plt.plot(prediction_plot)
plt.show()
