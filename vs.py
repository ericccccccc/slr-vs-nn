# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 22:04:55 2020

@author: Eric
"""
import numpy as np 
from scipy.stats import norm
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt


def crtCDF(x):
    if(type(x) == np.ndarray):
        loc = x.mean()
        scale = x.std()
        N = x.size
        pos = norm.cdf(x, loc, scale)*N
        return pos
    else:
        print("Wrong Type! x must be np.ndarray ~")    
        return

x = np.random.random_integers(10000,size=1000)
x = np.sort(x)
y = crtCDF(x)
norm_x = preprocessing.scale(x)    # 標準化: 零均值化


x = np.reshape(x,(-1,1))
model=LinearRegression()
model.fit(x,y)
pred_y = model.predict(x)

plt.title("Simple Linear Regression Model")
plt.plot(x, y, '.',label="Origin")
plt.plot(x, pred_y,'.',label="Model")
plt.legend()
plt.xlabel("Key")
plt.ylabel("Pred_Pos = CDF(Key)")
plt.show()


model  = Sequential()
model.add(Dense(8, input_dim=1, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1))

sgd=keras.optimizers.SGD(lr=0.0001)    # lr:學習率,可調參數
model.compile(loss="mse", optimizer=sgd, metrics=["mse"])
model.fit(norm_x, y, epochs=1000, batch_size=32, verbose=0)  # norm_x:訓練資料, y:訓練目標
pred_y = model.predict(norm_x)

plt.title("Neural Network 8x8 Model")
plt.plot(x, y, '.',label="Origin")
plt.plot(x, pred_y,'.',label="Model")
plt.legend()
plt.xlabel("Key")
plt.ylabel("Pred_Pos = CDF(Key)")
plt.show()


model  = Sequential()
model.add(Dense(16, input_dim=1, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(1))

sgd=keras.optimizers.SGD(lr=0.0001)    # lr:學習率,可調參數
model.compile(loss="mse", optimizer=sgd, metrics=["mse"])
model.fit(norm_x, y, epochs=1000, batch_size=32, verbose=0)  # norm_x:訓練資料, y:訓練目標
pred_y = model.predict(norm_x)

plt.title("Neural Network 16x16 Model")
plt.plot(x, y, '.',label="Origin")
plt.plot(x, pred_y,'.',label="Model")
plt.legend()
plt.xlabel("Key")
plt.ylabel("Pred_Pos = CDF(Key)")
plt.show()


model  = Sequential()
model.add(Dense(32, input_dim=1, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(1))

sgd=keras.optimizers.SGD(lr=0.0001)    # lr:學習率,可調參數
model.compile(loss="mse", optimizer=sgd, metrics=["mse"])
model.fit(norm_x, y, epochs=1000, batch_size=32, verbose=0)  # norm_x:訓練資料, y:訓練目標
pred_y = model.predict(norm_x)

plt.title("Neural Network 32x32 Model")
plt.plot(x, y, '.',label="Origin")
plt.plot(x, pred_y,'.',label="Model")
plt.legend()
plt.xlabel("Key")
plt.ylabel("Pred_Pos = CDF(Key)")
plt.show()
