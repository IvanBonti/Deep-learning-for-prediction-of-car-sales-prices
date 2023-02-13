import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

#Data import
cars_df = pd.read_csv("car data.csv")

#Data visualization 
#I choose to search for the correlation between the year of the car's manufacture and its price.
sns.scatterplot(x = 'Year' ,y = 'Present_Price' , data = cars_df)

#correlation
f, ax = plt.subplots(figsize = (20,20))
sns.heatmap(cars_df.corr(), annot = True)

#Data cleaning
selected_features = ['Year' ,'Selling_Price','Kms_Driven']

x = cars_df[selected_features]
#y is the variable we want to predict given the input data x.
y = cars_df['Present_Price']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

#Normalizing outputs.
y = y.values.reshape(-1,1) 
y_scaled = scaler.fit_transform(y)

#Training
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_scaled,y_scaled,test_size=0.25) 

#Definition of the model

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units = 100, activation = 'relu', input_shape = (3,)))
model.add(tf.keras.layers.Dense(units = 100, activation = 'relu'))
model.add(tf.keras.layers.Dense(units = 100, activation = 'relu'))
model.add(tf.keras.layers.Dense(units=1, activation='linear'))

model.summary()

#compilation
model.compile(optimizer='Adam', loss='mean_squared_error')

epochs_hist = model.fit(x_train, y_train, epochs = 100, batch_size = 50, validation_split=0.2)

#Model evaluation
epochs_hist.history.keys()

#Graph
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Progress of loss during model training')
plt.xlabel('Epochs')
plt.ylabel('Training and Validation loss')
plt.legend(['Training loss','Validation loss'])

#Prediction
#Definition of a car with its respective inputs: 'Year' ,'Selling_Price','Kms_Driven'
x_test_1 = np.array([[2014,3.35,27000]])

#Scaling data

scaler_1 = MinMaxScaler()
x_test_scaled_1 = scaler_1.fit_transform(x_test_1)

#Making a prediction 
y_predict_1 = model.predict(x_test_scaled_1)

#Reversing the price correctly
y_predict_1 = scaler.inverse_transform(y_predict_1)
