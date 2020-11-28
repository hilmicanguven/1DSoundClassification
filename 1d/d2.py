# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 08:44:35 2020

@author: ASELSAN-hilmi
title: 2D CNN KULLANIRAK SINIFLANDIRMA MODELIİ YAPILDI. ANCAK BURADA RAW DATA DEĞİLDE SPECTOGRAM GÖRÜNTÜLERİNE BAKILARAK EĞİTİM YAPILDI.
MODELİ DEĞERLDİRMEK İÇİN BİR FOKNSİYON YAZDIM. RETURN OLARAK ACCURACY DEĞERİNİ DÖNDÜRÜYOR. BU MAKALEYE GÖRE BİRÇOK EKSİĞİ VAR.
"""
import time
import keras
from keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D
from keras.models import Sequential
import librosa
import librosa.display
import numpy as np
import pandas as pd
import random
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

# Read Data
data = pd.read_csv('./UrbanSound8K/metadata/UrbanSound8K.csv')
print(data.head(5))

print("data shape", data.shape)

# Get data over 3 seconds long
print([data['end']-data['start']]) #get duration
valid_data = data[['slice_file_name', 'fold' ,'classID', 'class']][data['end']-data['start'] >= 0 ]
print(valid_data.shape)


valid_data['path'] = 'fold' + valid_data['fold'].astype('str') + '/' + valid_data['slice_file_name'].astype('str')

D = [] # Dataset
i=0
for row in valid_data.itertuples():
    i=i+1
    if i%50==0:
        print(i)
    y, sr = librosa.load('./UrbanSound8K/audio/' + row.path)  
    #print("lenght of y:  ",len(y))
    #print("y: ",y)
    ps = librosa.feature.melspectrogram(y=y, sr=sr)
   # print("ps: ",ps)
  #  print("lenght of ps:  ",len(ps[0]))
    if ps.shape != (128, 128): continue
    D.append( (ps, row.classID) )
    
    #print(type(D))
    #print(len(D))
       
    

print("Number of samples: ", len(D))

#print(D[8])
#print(len(D[0]))
dataset = D
random.shuffle(dataset)

train = dataset[:7000]
test = dataset[7000:]

X_train, y_train = zip(*train)
X_test, y_test = zip(*test)

print(type(X_train))
print(len(X_train))
print(X_train[0])

# Reshape for CNN input
X_train = np.array([x.reshape( (128*128, 1) ) for x in X_train])
X_test = np.array([x.reshape( (128*128, 1) ) for x in X_test])
print("**")
print("**")
print("**")
print(type(X_train))
print(X_train.shape)
print(X_train[0])
# One-Hot encoding for classes
y_train = np.array(keras.utils.to_categorical(y_train, 10))
y_test = np.array(keras.utils.to_categorical(y_test, 10))
print(type(y_train))
print(y_train.shape)
print(y_train[0])

model = Sequential()
input_shape=(128*128, 1)

model.add(Conv1D(16, (5), strides=(2), input_shape=input_shape))
model.add(MaxPooling1D((8), strides=(8)))
model.add(Activation('relu'))

model.add(Conv1D(32, (5), padding="valid"))
model.add(MaxPooling1D((8), strides=(8)))
model.add(Activation('relu'))

model.add(Conv1D(64, (5), padding="valid"))
model.add(Activation('relu'))

model.add(Conv1D(128, (5), padding="valid"))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dropout(rate=0.25))

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(rate=0.25))

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(rate=0.25))

model.add(Dense(10))
model.add(Activation('softmax'))

opt = keras.optimizers.Adadelta(learning_rate=1.0)
model.compile(
	optimizer=opt,
	loss="categorical_crossentropy",
	metrics=['accuracy'])

model.fit(
	x=X_train, 
	y=y_train,
    epochs=10,
    batch_size=100,
    validation_data= (X_test, y_test))

score = model.evaluate(
	x=X_test,
	y=y_test)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


