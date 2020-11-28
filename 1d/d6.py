
"""
Created on Thu Oct  1 14:38:50 2020

@author: ASELSAN-HİLMİ
TITLE: using 1DCNN sound classification RAW DATA 
valid_split= 0.2 ile yapıldı. VALIDATION DATA AYRI OLARAK VERİLMEDİ
GENEL İŞLEMLER D7.PY İLE AYNI. MODEL.CHECKPOINT KULLANILDI. SADECE VAL_ACCURACY ARTTIĞINDA KAYDEDİYOR.(AYRIYETEN SON MODELİ DE KAYDEDER)
MODELLERİ KAYDETTİĞİMİZ KLASÖRÜ, KODUN ÇALIŞTIĞI KLASÖRÜN İÇİNDE FARKLI BİR KLASÖR ADIYLA AÇMAK İYİ BİR FİKİR

"""
import time
import keras
from keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

import librosa
import librosa.display
import numpy as np
import pandas as pd
import random
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D






def overlap(y,SR,PO,classID,D):
    
    
    size = len(y)
    if 3*SR<size<=4*SR:
        add = 4*SR - lenght
        #print(add)
        b= y[0:add]
        y = np.concatenate((y,b))
        #print("2küsür", len(y))
        
        for i in range(0,4):
            y2 = y[(i*SR):(i*SR+SR)]
            D.append((y2, classID))
            if i<3:
                for j in range (1,int(1/PO)):
                    y3 = y[int(i*SR +int(j*SR*PO)) : int(i*SR+SR +int(j*SR*PO))  ]
                    D.append((y3, classID))
            
    

    
    elif 2*SR < lenght <= 3*SR: 
        add = 48000 - lenght
        #print(add)
        b= y[0:add]
        y = np.concatenate((y,b))
        #print("2küsür", len(y))
        
        for i in range(0,3):
            y2 = y[(i*SR):(i*SR+SR)]
            D.append((y2, classID))
            if i<2:
                for j in range (1,int(1/PO)):
                    y3 = y[int(i*SR + int(j*SR*PO)) : int(i*SR+SR +int(j*SR*PO))  ]
                    D.append((y3, classID))
            
            
  
    
    elif SR < lenght <= 2*SR:
        add = 32000 - lenght
        #print(add)
        b= y[0:add]
        y = np.concatenate((y,b))
        #print("1küsür", len(y))
        for i in range(0,2):
            y2 = y[(i*SR):(i*SR+SR)]
            D.append((y[(i*SR):(i*SR+SR)], classID))
            if i<1:
                for j in range (1,int(1/PO)):
                    y3 = y[int(i*SR +(j*SR*PO)) : int(i*SR+SR +(j*SR*PO))  ]
                    D.append((y[int(i*SR +(j*SR*PO)) : int(i*SR+SR +(j*SR*PO))  ], classID))
           
        

        
    elif lenght <= SR:
        y=tamamla(y)
        #print("0küsür", len(y))
        D.append( (y[0:SR], classID) )
     
    
    else:
        print("none")
        
    return D


def tamamla(y):
    while 0 < len(y) < 16000:
        eksik = 16000-len(y)
        #print("bu kadar eksik: ",eksik)
        b= y[0:eksik]
        y = np.concatenate((y, b))
        #print("yeni len uzunluğu: ",len(y))
        
    
    return y
# Read Data


data = pd.read_csv('./UrbanSound8K/metadata/UrbanSound8K.csv')
print(data.head(5))


#
# Get data over 3 seconds long
#print([data['end']-data['start']]) #get duration
valid_data = data[['slice_file_name', 'fold' ,'classID', 'class']][data['end']-data['start'] >= 0.0 ][data['fold'] <8]


valid_data['path'] = 'fold' + valid_data['fold'].astype('str') + '/' + valid_data['slice_file_name'].astype('str')
func = []
D = [] # Dataset
i=0
for row in valid_data.itertuples():
    i=i+1
    if i%100==0:
        print(i)
    
    #if i==13:
        #break
    y, sr = librosa.load('./UrbanSound8K/audio/' + row.path, sr=16000)  
    #print("lenght of y:  ",len(y))
    #print("y: ",type(y))
    #ps = librosa.feature.melspectrogram(y=y, sr=sr)
    #print("ps: ",ps)
    #print("lenght of ps:  ",len(ps[0]))
    lenght= len(y)
    #print(lenght)
    #print(y)
    overlap_percentage = 0.5
   
    D = (overlap(y,sr,overlap_percentage,row.classID,D))
    

    

print("Number of samples: ", len(D))
print("type of samples: ", type(D))

#print(D[8])
#print(len(D[0]))
dataset = D
random.shuffle(dataset)
print("len of dataset(TRAIN SET): ", len(dataset))

train = dataset

X_train, y_train = zip(*dataset) 
print("lenght of X_train and Y_train***",len(X_train), len(y_train))
#X_test, y_test = zip(*test)
"""
print(type(X_train))
print(len(X_train))
print(X_train[0])
"""
#print(X_test)
# Reshape for CNN input
X_train = np.array([x.reshape( (16000, 1) ) for x in X_train])
#X_test = np.array([x.reshape( (16000, 1) ) for x in X_test])
"""
print("**")
print("**")
print("**")
print(type(X_train))
print(X_train.shape)
print(X_train[0])
"""
# One-Hot encoding for classes
y_train = np.array(keras.utils.to_categorical(y_train, 10))
#y_test = np.array(keras.utils.to_categorical(y_test, 10))
"""
print(type(y_train))
print(y_train.shape)
print(y_train[0])
"""
model = Sequential()
input_shape=(16000, 1)

model.add(Conv1D(16, (64), strides=(2), input_shape=input_shape))
model.add(MaxPooling1D((8), strides=(8)))
model.add(Activation('relu'))

model.add(Conv1D(32, (32), strides=(2),padding="valid"))
model.add(MaxPooling1D((8), strides=(8)))
model.add(Activation('relu'))

model.add(Conv1D(64, (16),strides=(2), padding="valid"))
model.add(Activation('relu'))

model.add(Conv1D(128, (8), strides=(2),padding="valid"))
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

# checkpoint
filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(
	x=X_train, 
	y=y_train,
    epochs=100,
    batch_size=100,
    callbacks=callbacks_list,
    validation_split=0.2
    )

#score = model.evaluate(
#	x=X_test,
#	y=y_test)

#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

#print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
# save model and architecture to single file
model.save("h-model4.h5")
print("Saved model to disk")



