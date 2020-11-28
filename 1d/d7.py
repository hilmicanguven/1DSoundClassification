# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 10:20:11 2020

@author: ASELSAN-HİLMİ
TITLE : using 1DCNN sound classification RAW DATA 
        USE FOLD7=VALIDATION DATA AND FEED NN SEPERATELY
        USE FOLD0-1-2-3-4-5-6=TRAINING DATA
OUTPUT: IN SPECIFIC FOLDER, SAVE MODELS WITH MODEL CALLBACK. AND ALSO FINAL MODEL.
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





#IMPORTANT FUNCTION: ACCEPT SOUND FILE AND SPLIT INTO FRAMES AND ADD DATASET. ALSO MAKE OVERLAPPING
#IF FILE 2.5 SEC, THEN MAKE THEM 3SEC
def overlap(y,SR,PO,classID,D):   
    
    
    lenght = len(y)
    if 3*SR<lenght<=4*SR:
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

#IF SOUND SHORTER THAN 1 SEC, WE ADD THIS ITSELF AND COMPLETE 1 SEC
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
# show first 5 data 
print(data.head(5))


#print([data['end']-data['start']]) # get duration longer then some seconds, now longer than 0
#fold7 will be validation dataset
validation_data = data[['slice_file_name', 'fold' ,'classID', 'class']][data['end']-data['start'] >= 0.0 ][data['fold'] == 7]

#fold0-1-2-3-4-5-6 will be trainig dataset(valid_data)
valid_data = data[['slice_file_name', 'fold' ,'classID', 'class']][data['end']-data['start'] >= 0.0 ][data['fold'] <7]


valid_data['path'] = 'fold' + valid_data['fold'].astype('str') + '/' + valid_data['slice_file_name'].astype('str')
validation_data['path'] = 'fold' + validation_data['fold'].astype('str') + '/' + validation_data['slice_file_name'].astype('str')

D_validation = [] #dataset for validation
D = [] # Dataset for training
i=0
j=0

for row in valid_data.itertuples():   #reach and get every wav file
    #every 100 sample, print
    i=i+1
    if i%100==0:
        print(i)
    
    #if i==13:
        #break
    #GET SOUND FILE AS A RAW DATA. CHOOSE SAMPLING RATE=16000. ALSO DECIDE OVERLAP PERCENTAGE, %25 %50 OR STH
    y, sr = librosa.load('./UrbanSound8K/audio/' + row.path, sr=16000)  
    overlap_percentage = 0.5
    #I BUILD A FUNCTION CALLED "OVERLAP". IT DOES: ACCEPT Y:DATA, SR:SAMPLING RATE, OVERLAP PERCENTAGE, CLASS OF EACH FILE, DATASET OF TRAINING 
    D = (overlap(y,sr,overlap_percentage,row.classID,D))  #FUNCTION RETURNS DATASET (IT IS RESULTANT DATASET CONSIST OF EACH FILE WITH OVERLAPPPING FRAMES)


#SAME PROCEDURE FOR VALIDATION SET    
for row_validation in validation_data.itertuples():
    y, sr = librosa.load('./UrbanSound8K/audio/' + row_validation.path, sr=16000)  
    overlap_percentage = 0.5
    D_validation = (overlap(y,sr,overlap_percentage,row_validation.classID,D_validation))
    #if j==13:
        #break
    j +=1

print("Number of train samples: ", len(D))
print("Number of validation samples: ", len(D_validation))


dataset = D
#IN ORDER TO MAKE RANDOMNESS, USE SUFFLE. THEN EVERY DATA SHOULD BE MIXED
random.shuffle(dataset)
print("len of dataset(TRAIN SET): ", len(dataset))
train = dataset
#NOW DIVIDE DATA AND ITS CLASS. X_TRAIN USED FOR FEEDING NEURAL NETWORK ACCORDING TO ITS OUTPUT(CLASS_ID)
X_train, y_train = zip(*dataset) 
print("lenght of X_train and Y_train***",len(X_train), len(y_train))
#X_test, y_test = zip(*test)
# Reshape for CNN input
X_train = np.array([x.reshape( (16000, 1) ) for x in X_train])  #SINCE WE BUILD NN TAKES 1SEC INTERVAL OF EACH FILE(CORRESPONDS TO 16000 LENGHT OF ARRAY)
#X_test = np.array([x.reshape( (16000, 1) ) for x in X_test])   #RESHAPE 16000,1 TO AVOID DIMENSION ERRORS

# One-Hot encoding for classes
y_train = np.array(keras.utils.to_categorical(y_train, 10))  #WE HAVE 10 CLASSES AND CONVERT THEM TO 0-1-2-...-9
#y_test = np.array(keras.utils.to_categorical(y_test, 10))


#SAME PROCEDURE FOR VALIDATION SET
dataset_validation = D_validation
random.shuffle(dataset_validation)
validation = dataset_validation
X_val, y_val = zip(*validation)
X_val = np.array([x.reshape( (16000,1) ) for x in X_val])
y_val = np.array(keras.utils.to_categorical(y_val, 10))

#MOST IMPORTANT PART-> BUILD ARCHITECTURE OF NEURAL NETWORK (CONVOLUTIONAL)
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

# checkpoint USED FOR : SAVE MODEL IF THERE IS ANY IMPROVMENT OCCUR IN VALIDATION ACCURACY
filepath="./02_10modelcallback4/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(
	x=X_train, 
	y=y_train,
    epochs=100,
    batch_size=100,
    callbacks=callbacks_list,
    shuffle=True,
    validation_data = (X_val, y_val)
    )


#UP TO THIS POINT, BUILD NEURAL NETWORK + OPTIMIZERS + FIT IT(TRAIN PARAMETERS)

#score = model.evaluate(
#	x=X_test,
#	y=y_test)

#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

#print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

# save model and architecture to single file
model.save("h-model6_0210.h5")
print("Saved model to disk")



