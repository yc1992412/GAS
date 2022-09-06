# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 22:25:07 2020
@author: yuancheng
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

pi = np.pi

#%% 精确解

x = np.arange(0,1,0.01)
y = np.arange(0,1,0.01)

X,Y = np.meshgrid(x,y)
Z = np.cos(pi*X)*np.cos(pi*Y)

fig, ax = plt.subplots()
ax.pcolor(X,Y,Z,cmap='jet')

#%% Making data

def sampling(batch = 64):
    
    
    batch_bd = int(np.sqrt(batch))
    
    
    b1 = np.zeros((batch_bd, 2))
    b2 = np.zeros((batch_bd, 2))
    b3 = np.zeros((batch_bd, 2))
    b4 = np.zeros((batch_bd, 2))
    
    b1[:,1] = 0
    b1[:,0] = np.random.rand(batch_bd)
    
    b2[:,1] = 1
    b2[:,0] = np.random.rand(batch_bd)
    
    b3[:,0] = 0
    b3[:,1] = np.random.rand(batch_bd)
    
    b4[:,0] = 1
    b4[:,1] = np.random.rand(batch_bd)
    
    bd_data = np.vstack([b1,b2,b3,b4])
    
    num_inner_point = int((batch))
    
    in_data = np.random.rand(num_inner_point,2)
    
    f_data = np.zeros((num_inner_point,1))
    f_data[:,0] = np.cos(pi*in_data[:,0])*np.cos(pi*in_data[:,1])*2*pi**2
    
    return [np.float32(bd_data), np.float32(in_data), np.float32(f_data)]

#%%
inner = keras.Input(shape=(2,))
boundary = keras.Input(shape=(2,))
f = keras.Input(shape=(1,))

H1 = layers.Dense(50, activation='relu')
H2 = layers.Dense(100, activation='relu')
H3 = layers.Dense(100, activation='relu')
H4 = layers.Dense(50, activation='relu')
H5 = layers.Dense(1, activation='linear', name = 'u')

with tf.GradientTape() as tape:
    tape.watch(inner)
    h1 = H1(inner)
    h2 = H2(h1)
    h3 = H3(h2)
    h4 = H4(h3)
    u = H5(h4)

du = tape.gradient(u,inner)
ux = du[:,0:1]
uy = du[:,1:]

energy = (ux**2+uy**2)/2-f*u
integral = (K.mean(u))

model = keras.Model(inputs = [inner, f], outputs = [energy,integral])
#%%
def Innerloss(y_pred):
    loss = K.mean(y_pred)
    return loss

def Boundloss(y_pred):
    loss = 200*K.abs(y_pred)
    return loss

batch_size = 10000
optimizer = keras.optimizers.Adam()

history = []

steps = 5000

for step in range(steps):

    [bd_data, in_data, f_data] = sampling(batch_size)

    with tf.GradientTape() as tape:
        [energy,integral] = model([in_data, f_data], training=True)
        loss1 = Innerloss(energy)
        loss2 = Boundloss(integral)
        loss = loss1 + loss2     

    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    if step%50 ==0:
        print('step = ', step,
              'loss_in = ', float(loss1),'loss_bd = ', float(loss2))
        history.append([step, float(loss), float(loss2)])
#%%
Z1 = 0*Z
model_u = keras.Model(inputs = inner, outputs = model.get_layer('u').output)

xy = []

for i in tqdm(range(len(x))):
    for j in range(len(y)):
        xy.append(np.array([x[i],y[j]]))

xy = np.array(xy)
u = model_u.predict(xy)
Z1 = u.reshape((100,100))


plt.figure(2)
history = np.array(history)
plt.plot(history[:,0], history[:,1])

plt.figure(1)
fig1, ax1 = plt.subplots()
ax1.pcolor(Y,X,Z1,cmap='jet')





















