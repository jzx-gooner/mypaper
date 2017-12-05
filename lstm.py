from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

# define a function to convert a vector of time series into a 2D matrix
def convertSeriesToMatrix(vectorSeries, sequence_length):
    matrix=[]
    for i in range(len(vectorSeries)-sequence_length+1):
        matrix.append(vectorSeries[i:i+sequence_length])
    return matrix
    
# random seed
np.random.seed(1234)
   
# wind_speed raw data
df_raw = pd.read_csv('hourly_wind_speed_2016.csv', header=None)
# numpy array
df_raw_array = df_raw.values
# daily wind_speed
list_daily_wind_speed = [df_raw_array[i,:] for i in range(0, len(df_raw)) if i % 24 == 0]
# hourly wind_speed (23 wind_speeds for each day)
list_hourly_wind_speed = [df_raw_array[i,1]/100000 for i in range(0, len(df_raw)) if i % 24 != 0]
# the length of the sequnce for predicting the future value
sequence_length = 23

# convert the vector to a 2D matri
matrix_wind_speed = convertSeriesToMatrix(list_hourly_wind_speed, sequence_length)

# shift all data by mean
matrix_wind_speed = np.array(matrix_wind_speed)
shifted_value = matrix_wind_speed.mean()
matrix_wind_speed -= shifted_value
print ("Data  shape: ", matrix_wind_speed.shape)

# split dataset: 90% for training and 10% for testing
train_row = int(round(0.9 * matrix_wind_speed.shape[0]))
train_set = matrix_wind_speed[:train_row, :]

# shuffle the training set (but do not shuffle the test set)
np.random.shuffle(train_set)
# the training set
X_train = train_set[:, :-1]
# the last column is the true value to compute the mean-squared-error loss
y_train = train_set[:, -1] 
# the test set
X_test = matrix_wind_speed[train_row:, :-1]
y_test = matrix_wind_speed[train_row:, -1]

# the input to LSTM layer needs to have the shape of (number of samples, the dimension of each element)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 搭建神经网络 2层lstm，一层全链接层
model = Sequential()
# layer 1: LSTM
model.add(LSTM( input_dim=1, output_dim=50, return_sequences=True))
model.add(Dropout(0.2))
# layer 2: LSTM
model.add(LSTM(output_dim=100, return_sequences=False))
model.add(Dropout(0.2))
# layer 3: dense
# linear activation: a(x) = x
model.add(Dense(output_dim=1, activation='linear'))
# compile the model
model.compile(loss="mse", optimizer="rmsprop")

# train the model
model.fit(X_train, y_train, batch_size=512, nb_epoch=50, validation_split=0.05, verbose=1)

# evaluate the result
test_mse = model.evaluate(X_test, y_test, verbose=1)
print ('\nThe mean squared error (MSE) on the test data set is %.3f over %d test samples.' % (test_mse, len(y_test)))

# get the predicted values
predicted_values = model.predict(X_test)
num_test_samples = len(predicted_values)
predicted_values = np.reshape(predicted_values, (num_test_samples,1))

# plot the results
fig = plt.figure()
plt.plot(y_test + shifted_value)
plt.plot(predicted_values + shifted_value)
plt.xlabel('times')
plt.ylabel('wind_speed (M/s)')
plt.show()
fig.savefig('output_wind_speed_forecasting.jpg', bbox_inches='tight')

# save the result into txt file
test_result = np.vstack((predicted_values, y_test)) + shifted_value
np.savetxt('output_wind_speed_forecasting_result.txt', test_result)
