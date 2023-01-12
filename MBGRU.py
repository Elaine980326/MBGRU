# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 16:26:25 2022

@author: 54946
"""
import numpy as np
import scipy.io as io
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import time
import pandas as pd

## 井数据每列参数为：depth |DT(P波时差) GR(伽马射线) SP（自发电位） DEN（密度） | PERM（渗透率） POR（孔隙度） SW（含水饱和度） VSH（泥质含量）
well_1_=io.loadmat('nb1.mat')
well_1=well_1_.get('nb1').astype(np.float32)

well_2_=io.loadmat('nb4.mat')
well_2=well_2_.get('nb4').astype(np.float32)

x_train=well_1[:,1:5]
y_train=well_1[:,[6,8]]
x_test=well_2[:,1:5]
y_test=well_2[:,[6,8]]

depth1=well_1[:,0]
depth2=well_2[:,0]

from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(x_train) 
scaled_x_train = scaler.transform(x_train)   
scaled_x_test = scaler.transform(x_test)     

scaler_label = preprocessing.StandardScaler().fit(y_train)
scaled_y_train=scaler_label.transform(y_train)
scaled_y_test=scaler_label.transform(y_test)

model=tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(4,1)),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64)),
    tf.keras.layers.Reshape((-1,1)),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64)),
    tf.keras.layers.Reshape((-1,1)),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64)),
    tf.keras.layers.Dropout(0.5),
    # tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(2)
])


model.summary()

model.compile(loss='mse',
                # optimizer=optimizer,
                optimizer='Adamax',
                metrics=['mae', 'mse'])

EPOCHS = 150

time_start=time.time()
history = model.fit(
  scaled_x_train,scaled_y_train,
  epochs=EPOCHS, verbose=1,
   validation_split=0.2
  # validation_data=(scaled_x_test,scaled_y_test)
  # callbacks=[PrintDot()]
  )

time_end=time.time()
print('time cost:',(time_end-time_start)/60,'min')

model_GRU=tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(4,1)),
    # tf.keras.layers.Dropout(0.5),
    # tf.keras.layers.SimpleRNN(64),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    # tf.keras.layers.LSTM(64),
    tf.keras.layers.GRU(64,dropout=0),
    # tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2)
])

model_GRU.summary()

model_GRU.compile(loss='mse',
                # optimizer=optimizer,
                optimizer='Adamax',
                metrics=['mae', 'mse'])

history_GRU = model_GRU.fit(
  scaled_x_train,scaled_y_train,
  epochs=EPOCHS, verbose=1,
   validation_split=0.2
  # validation_data=(scaled_x_test,scaled_y_test)
  # callbacks=[PrintDot()]
  )

y_train_pred0=model.predict(scaled_x_train)
y_train_pred=scaler_label.inverse_transform(y_train_pred0)
y_test_pred0= model.predict(scaled_x_test)
y_test_pred=scaler_label.inverse_transform(y_test_pred0)

y_test_pred_GRU= model_GRU.predict(scaled_x_test)
y_test_pred_GRU=scaler_label.inverse_transform(y_test_pred_GRU)



hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

hist_GRU = pd.DataFrame(history_GRU.history)
hist_GRU['epoch'] = history_GRU.epoch
hist_GRU.tail()

mean_mae_MBGRU=hist['val_mae'][10:].sum()/(EPOCHS-10)
mean_mae_GRU=hist_GRU['val_mae'][10:].sum()/(EPOCHS-10)
mean_mse_MBGRU=hist['val_mse'][10:].sum()/(EPOCHS-10)
mean_mse_GRU=hist_GRU['val_mse'][10:].sum()/(EPOCHS-10)

print('mean val_mae of MBGRU :',mean_mae_MBGRU)
print('mean val_mae of GRU :',mean_mae_GRU)
print('mean val_mse of MBGRU :',mean_mse_MBGRU)
print('mean val_mse of GRU :',mean_mse_GRU)

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  # plt.figure()
  # plt.xlabel('Epoch')
  # plt.ylabel('loss')
  # plt.plot(hist['epoch'], hist['loss'],
  #          label='Train loss')
  # plt.plot(hist['epoch'], hist['val_loss'],
  #          label = 'Val loss')
  # # plt.ylim([0,0.1])
  # plt.legend()
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0,0.8])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0,0.5])
  plt.legend()
  plt.show()

plot_history(history)
plot_history(history_GRU)


f, ax = plt.subplots(nrows=1, ncols=2,  figsize=(9, 15))
ax[0].plot(y_train[:,0]*100,depth1,color='black',label='True')
ax[0].plot(y_train_pred[:,0]*100,depth1, '--', color='r',label='Pred')
ax[0].invert_yaxis()
ax[0].set_ylabel('Depth(m)')
ax[0].set_xlabel('Porosity(%)')
ax[0].set_ylim([3020,2780])
ax[0].set_xlim([min(y_train[:,0]*100)-0.1,max(y_train[:,0]*100)+0.1])

#ax[1].plot(y_train[:,1],depth1,color='black',label='True')
#ax[1].plot(y_train_pred[:,1],depth1, '--', color='r',label='Pred')
#ax[1].invert_yaxis()
#ax[1].set_ylabel('depth(ft)')
#ax[1].set_xlabel('SW(%)')
#ax[1].set_ylim([2275,2085])
#ax[1].set_xlim([min(y_train[:,1]-0.1),max(y_train[:,1])+0.1])


ax[1].plot(y_train[:,1]*100,depth1,color='black',label='True')
ax[1].plot(y_train_pred[:,1]*100,depth1, '--', color='r',label='Pred')
ax[1].invert_yaxis()
# ax[1].set_ylabel('depth(ft)')
ax[1].set_xlabel('Vsh(%)')
ax[1].set_ylim([3020,2780])
ax[1].set_xlim([min(y_train[:,1]*100)-0.1,max(y_train[:,1]*100)+0.1])

ax[1].set_yticklabels([]);

ax[1].set_yticklabels([]); #ax[2].set_yticklabels([])
ax[0].legend();ax[1].legend();#ax[2].legend()
plt.show()

f1, ax1 = plt.subplots(nrows=1, ncols=4, figsize=(9, 15))
ax1[0].plot(y_test[:,0]*100,depth2,color='black',label='True')
ax1[0].plot(y_test_pred[:,0]*100,depth2, '--', color='r',label='Pred')
ax1[0].invert_yaxis()
# ax1[0].set_xlim(-0.3,11)
ax1[0].set_ylabel('Depth(m)')
ax1[0].set_xlabel('Porosity(%)')
ax1[0].set_ylim([max(depth2),min(depth2)-15])
ax1[0].set_xlim([min(y_test[:,0]*100)-0.1,max(y_test[:,0]*100)+0.1])
ax1[0].set_title('BMGRU')

ax1[1].plot(y_test[:,0]*100,depth2,color='black',label='True')
ax1[1].plot(y_test_pred_GRU[:,0]*100,depth2, '--', color='r',label='Pred')
ax1[1].invert_yaxis()
# ax[1].set_ylabel('depth(ft)')
ax1[1].set_xlabel('Porosity(%)')
ax1[1].set_ylim([max(depth2),min(depth2)-15])
ax1[1].set_xlim([min(y_test[:,0]*100)-0.1,max(y_test[:,0]*100)+0.1])
ax1[1].set_title('GRU')

ax1[2].plot(y_test[:,1]*100,depth2,color='black',label='True')
ax1[2].plot(y_test_pred[:,1]*100,depth2, '--', color='r',label='Pred')
ax1[2].invert_yaxis()
# ax[2].set_ylabel('depth(ft)')
ax1[2].set_xlabel('Vsh(%)')
ax1[2].set_ylim([max(depth2),min(depth2)-15])
ax1[2].set_xlim([min(y_test[:,1]*100)-0.1,max(y_test[:,1]*100)+0.1])
ax1[2].set_title('BMGRU')

ax1[3].plot(y_test[:,1]*100,depth2,color='black',label='True')
ax1[3].plot(y_test_pred_GRU[:,1]*100,depth2, '--', color='r',label='Pred')
ax1[3].invert_yaxis()
# ax[3].set_ylabel('depth(ft)')
ax1[3].set_xlabel('Vsh(%)')
ax1[3].set_ylim([max(depth2),min(depth2)-15])
ax1[3].set_xlim([min(y_test[:,1]*100)-0.1,max(y_test[:,1]*100)+0.1])
ax1[3].set_title('GRU')


ax1[1].set_yticklabels([]); #ax1[2].set_yticklabels([])
ax1[2].set_yticklabels([]);
ax1[3].set_yticklabels([]);
ax1[0].legend();ax1[1].legend();#ax1[2].legend()
ax1[2].legend();ax1[3].legend();
plt.show()

y_test_small=y_test[0:-1:3]*100
y_test_pred_small=y_test_pred[0:-1:3]*100
fig1, ax2 = plt.subplots()
ax2.scatter(y_test_small[:,0], y_test_pred_small[:,0], edgecolors=(0,0,0)) #画散点图
ax2.plot([y_test_small.min(), y_test_small.max()], [y_test_small.min(), y_test_small.max()], "k--", lw=4)
ax2.set_xlabel("Por Actual(%)")
ax2.set_ylabel("Por Predicted(%)")
ax2.set_xlim([0, 20])
ax2.set_ylim([0, 20])
plt.show()

fig2, ax3 = plt.subplots()
ax3.scatter(y_test_small[:,1], y_test_pred_small[:,1], edgecolors=(0,0,0)) #画散点图
ax3.plot([y_test_small.min(), y_test_small.max()], [y_test_small.min(), y_test_small.max()], "k--", lw=4)
ax3.set_xlabel("VSH Actual(%)")
ax3.set_ylabel("VSH Predicted(%)")
ax3.set_xlim([0, 100])
ax3.set_ylim([0, 100])
plt.show()

