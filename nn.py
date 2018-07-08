#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 09:55:42 2018

@author: ZIYU
"""

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

infile = open('final.csv','r')
data = infile.readlines()[1:]
all_list = []

for l in data:
    data_l = l.strip('\n').split(',')
    all_list.append(data_l)
#for i in range(3):
#    print(all_list[i])

def get_month_day_time(data_list):
    col1 = np.zeros((len(data_list),1),dtype = int)
    col2 = np.zeros((len(data_list),1),dtype = int)
    col3 = np.zeros((len(data_list),1),dtype = int)
    for i in range(len(data_list)):
        l1 = data_list[i][0].split()
        l2 = l1[0].split('-')
        col1[i][0] = int(l2[1])
        col2[i][0] = int(l2[2])
        l3 = l1[1].split(':')
        d = int(l3[0])
        if d >= 0 and d < 6:
            col3[i][0] = 0
        elif d >= 6 and d < 12:
            col3[i][0] = 1
        elif d >= 12 and d < 18:
            col3[i][0] = 2
        elif d >= 18 and d < 24:
            col3[i][0] = 3
    return np.concatenate((col1,col2,col3),axis = 1)
def get_vendor_lon1_lat1_lon2_lat2_isPeak_isHol(data_list):
    col4 = np.zeros((len(data_list),1),dtype = int)  #vendor
    col5 = np.zeros((len(data_list),1),dtype = float)#lon1
    col6 = np.zeros((len(data_list),1),dtype = float)#lat1
    col7 = np.zeros((len(data_list),1),dtype = float)#lon2
    col8 = np.zeros((len(data_list),1),dtype = float)#lat2
    col9 = np.zeros((len(data_list),1),dtype = int)#isPeak
    col10 = np.zeros((len(data_list),1),dtype = int)#isHol
    for i in range(len(data_list)):
        col4[i][0] = int(data_list[i][2])
        col5[i][0] = float(data_list[i][6])
        col6[i][0] = float(data_list[i][7])
        col7[i][0] = float(data_list[i][8])
        col8[i][0]= float(data_list[i][9])
        col9[i][0] = int(data_list[i][14])
        col10[i][0] = int(data_list[i][15])
    return np.concatenate((col4,col5,col6,col7,col8,col9,col10),axis = 1)
def get_labels(data_list):
    label = np.zeros((len(data_list),1),dtype = float)
    for i in range(len(data_list)):
        label[i][0] = float(data_list[i][10])
    return label 


#col1 = get_month(all_list)  
#print(col1.shape) #(855439,)
trunk1 = get_month_day_time(all_list)
#print(col2.shape) #(855439, 2)
trunk2 = get_vendor_lon1_lat1_lon2_lat2_isPeak_isHol(all_list)
#print(trunk2[0]) #(855439,7)
matrix = np.concatenate((trunk1,trunk2),axis = 1)
#print(len(matrix))
#print(matrix.shape)
#print(matrix)
mean = np.mean(matrix, axis = 0)
std = np.std(matrix, axis = 0)
import pickle
archive = {'mean':mean,'std':std}
outfile = open('archive.dat','wb')
pickle.dump(archive,outfile)
outfile.close()

nor_matrix = (matrix-mean)/std
#print(np.mean(matrix, axis = 0))
#print(np.std(matrix, axis = 0))

label = get_labels(all_list)
#print(label.shape)
train_X,test_X,train_Y,test_Y = train_test_split(nor_matrix,label,test_size = 0.1,random_state=1)
#print(train_X.shape)
#print(test_X.shape)
#print(train_Y.shape)

model = Sequential()
model.add(Dense(128,activation = 'relu',input_shape=(10,)))
model.add(Dense(256,activation = 'relu'))
model.add(Dense(256,activation = 'relu'))
model.add(Dense(128,activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(1))
model.compile(loss = 'mean_squared_logarithmic_error',optimizer = Adam(lr=0.001, decay=1e-6), metrics = None)

#model = load_model('my_model_7.h5')

history = model.fit(train_X,train_Y,batch_size = 1024, shuffle = True, epochs = 50,\
                  validation_data = (test_X,test_Y),\
                  callbacks = [EarlyStopping(min_delta = 0.000001, patience = 5)])
model.save('my_model_10.h5')
print(history.history.keys())
image_count = 0
#image_count+=1
#plt.figure(image_count)
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
#plt.show()
image_count+=1
plt.figure(image_count)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
'''
model = load_model('my_model_7.h5')
pre = model.predict_on_batch(nor_matrix)
print(pre.shape)
‘’‘
