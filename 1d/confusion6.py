# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 07:48:10 2020

@author: ASELSAN
title: modelin başarısın ölçmek için confusion matrix iyi bir yöntemdir.bunun için bu kod hazırlanmıştır. daha sonra bu kodun çıktısı
util.ipynb dosyasnda kullanılıp daha güzel bir gösterim elde edilmiştir.kodun neler yaptığını satırlarda açıklayalım


"""


import matplotlib as plt
from matplotlib import figure
from statistics import mode
from scipy import stats as s

import seaborn as sn
from keras.models import load_model
import numpy as np
import librosa
# load and evaluate a saved model
from numpy import loadtxt
from keras.models import load_model
import pandas as pd

# summarize model.
#model.summary()
import os
y_pred = []   #tahmin edilen değerleri tutan array
y_true = []   #gerçek değerleri tutan array. bu ikisini index by index kıyaslayarak doğru veya yanlış bulabiliriz
y_pred2 = []
y_true2 = []
i=0


# load model
model = load_model('weights-improvement-10-0.58.hdf5')   #ÖLÇMEK İSTEDİĞİMİZ MODELİ LOAD EDERİZ

#bu fonksiyonu bir saniyeden az olan sesleri 1 saniyeye tamamlamak için oluşturduk. kendisiyle toplayarak
def tamamla(y):
    while 0 < len(y) < 16000:
        eksik = 16000-len(y)
        #print("bu kadar eksik: ",eksik)
        b= y[0:eksik]
        y = np.concatenate((y, b))        
    
    
    return y

#label fonksiyonu önemli bir fonksiyon. conf_matrix oluştururken elimizdeki verileri tahmin edip gerçek label'lerı ile karşılaştırmak gerekiyordu
#burada ilk adım olarak sesleri yine 1sec aralıklarında ayrı frame'lere ayırdık ve her biri için tahminde bulunduk. örneğin 4 saniye uzunluğundaki
#bir ses için her bir saniye aralığına bir adet olmak üzere toplam 7 adet(overlap olduğu için fazladan 3 adet daha tahmin geldi.)
#bu tahminleri daha sonradan aggregate etmemiz gerekiyordu.benim seçtiğim yöntem "majority voting" yani en fazla tahminde bulunulan sınıfı final_label olarak  seçeriz(7 tanesinden en fazla hangi sınıf var)
#☺bu fonksiyon bize bir ses dosyası için hangi label tahmininde bulundu onu gönderir. daha sonra da y_pred adlı arraye aktarılır.
def label(y,sr,classID1):
    list1 = []
    list2 = []
    labelled1 = 0
    size = len(y)
    if 3*sr<size<=4*sr:
        add = 64000-size
        b= y[0:add]
        y = np.concatenate((y,b)) #y=4s now
        
        for i in range(0,4):
            y2 = y[(i*sr):(i*sr+16000)]
            y2=np.reshape(y2, (1,16000,1))
            a1 = model.predict_classes(y2)
            list1.append(int(a1))
            list2.append(int(classID1))
            if i<3:
                y2 = y[(i*sr+8000):(i*sr+16000+8000)]
                y2=np.reshape(y2, (1,16000,1))
                a1 = model.predict_classes(y2)
                list1.append(int(a1))
                list2.append(int(classID1))
            
        labelled1 = int(s.mode(list1)[0])
    
    elif 3*sr <  size <= 4*sr:
        add = 48000-size
        b= y[0:add]
        y = np.concatenate((y,b)) #y is exactly 3s now
        
        for i in range(0,3):
            y2 = y[(i*sr):(i*sr+16000)]
            y2=np.reshape(y2, (1,16000,1))
            a1 = model.predict_classes(y2)
            list1.append(int(a1))
            list2.append(int(classID1))
            if i<2:
                y2 = y[(i*sr+8000):(i*sr+16000+8000)]
                y2=np.reshape(y2, (1,16000,1))
                a1 = model.predict_classes(y2)
                list1.append(int(a1))
                list2.append(int(classID1))
                
            
        labelled1 = int(s.mode(list1)[0])
        
    elif sr <  size <= 2*sr:
        add = 32000-size
        b= y[0:add]
        y = np.concatenate((y,b)) #y is exactly 2s now
        
        for i in range(0,2):
            y2 = y[(i*16000):((i*16000)+16000)]
            y2=np.reshape(y2, (1,16000,1))
            a1 = model.predict_classes(y2)
            list1.append(int(a1))
            list2.append(int(classID1))
            if i<1:
                y2 = y[(i*sr+8000):(i*sr+16000+8000)]
                y2=np.reshape(y2, (1,16000,1))
                a1 = model.predict_classes(y2)
                list1.append(int(a1))
                list2.append(int(classID1))
            
            
        labelled1 = int(s.mode(list1)[0])
    return labelled1
    #print("y_pred inside func: ",y_pred1)
    #print("y_pred inside func: ",y_true1)
    return labelled1


#####   MAIN   FUNCTION    ######
df = pd.read_csv('./UrbanSound8K/metadata/UrbanSound8K.csv')
df = df[['slice_file_name', 'fold' ,'classID', 'class']][df['end']-df['start'] >= 0.0 ][df['fold'] > 7]
df['path'] = 'fold' + df['fold'].astype('str') + '/' + df['slice_file_name'].astype('str')
final_label = 0
#print('I love {} for "{}!"'.format('Geeks', 'Geeks'))


for row in df.itertuples():
    y, sr = librosa.load('./UrbanSound8K/audio_yedek/' + row.path, sr=16000)    
    #print("raw data: ",row.path)
    #TAHMİNLERİ 2 AYRI KATEGORİDE YAPARIZ. 1saniyeden küçük olanlar ve byük olanlar için
    if len(y)<= 16000:
        #print(len(y))
        y = tamamla(y)
        #print(len(y))
        y = y[0:16000]
        y=np.reshape(y, (1,16000,1))
        a = model.predict_classes(y)
        y_pred2.append(int(a[0]))
        y_true2.append(int(row.classID))
    else:
        final_label = label(y,sr,row.classID)
        y_pred2.append(int(final_label))
        y_true2.append(int(row.classID))
    
    i += 1
    if i%50==0:
        print(i)
print("i--",i)           
j=0
dogru= 0

#değerleri kendimiz karşılaştırarak doğru bilinen ses sayısını buluruz
for j in range(0,len(y_pred2)):
    if y_pred2[j]==y_true2[j]:
        dogru += 1
print("dogru", dogru)


print("dogru bilme oranı: ", dogru/len(y_pred2))
print(y_pred2)
print(y_true2)

from sklearn.metrics import confusion_matrix
#HAZIR BİR MODEL KULLANIP CONF MATRIX OLUŞTURURUZ. BU BİZE (class,class) BOYUTUNDA BİR MATRİS DÖNDÜRÜR.BUNUN SONUCUNDA OLAN MATRİSİ UTİLS.İPYNB DOSYASINA KOPYALADIK VE ORADA DAHA GÜZEL BİR GÖRÜNÜM ELDE ETTİK

array = confusion_matrix(y_true2, y_pred2)
print(array)

tp=0
tn=0
fp=0
fn=0

 #BİR FONK DAHA: BURADA DA BİZDEN İSTENİLEN BAZI METRIC DEĞERLRİNİ HESAPLARZ. BUNLAR   precision, recall, f1 SKORU
 #HER BİR METRIC İÇİN BELİRLİ FORMALİZAASYONLAR VARDI VE BU FONKSİOYNU DA KENDİM OLUŞTURDUM.
 # FONK RETURN DEĞERLERİ OLARAK precision, recall, f1 SKORU DÖNDÜRÜR.
def metric(y_pred2, y_true2,classID):   
    tp=0
    tn=0
    fp=0
    fn=0
    for t in range(0,len(y_pred2)):
        if y_pred2[t]==classID:
            if y_true2[t]==classID:
                tp += 1
            else:
                fp +=1
        else:
            if y_true2[t]==classID:
                fn +=1
            else:
                tn += 1
    
    #print("true positive: ",tp)
    #print("true negative: ",tn)
    #print("false positive: ",fp)
    #print("false negative: ",fn)
    
    precision = tp/(tp+fp)
    #print("precision: ",precision)
    
    recall = tp/(tp+fn)
    #print("precision: ",recall)
    
    f1 = (2*tp)/(2*tp+fp+fn)
    #print("precision: ",f1)
    return precision,recall,f1


precision=0
recall=0
f1=0

for i in range(0,10):
    precision, recall, f1=metric(y_pred2, y_true2, i)
    print("class: %d --- precision: %f " %(i,precision))
    print("class: %d --- recall %f: " %(i,recall))
    print("class: %d --- f1 skor %f: " %(i,f1))

        
"""
df_cm = pd.DataFrame(array, index = [i for i in "ABCDEFGHIJ"],
                  columns = [i for i in "ABCDEFGHIJ"])
figure.Figure(figsize = (10,10))
sn.heatmap(df_cm, annot=True)


"""