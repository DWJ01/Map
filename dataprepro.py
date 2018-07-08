#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 18:42:02 2018

@author: ZIYU
"""

import pickle
infile = open('archive.dat','rb')
dic = pickle.load(infile)
mean = dic['mean']
#print(mean.shape)
std = dic['std']
#print(std.shape)

from math import floor
import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
from IPython.display import display
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

earth_radius = 6371

def readData(filename):
    data = pd.read_csv(filename, parse_dates=[2,3])
    #display(data.head(3))
    return data
'''
def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

def set_peak(data):
    data.set_index('pickup_datetime', inplace=True, drop=False, append=False)
    data['pickup_hour'] = data.index.hour
    data['isPeak'] = (data['pickup_hour'].isin([7,8,9,18,19,20,21])).astype(int)
    
def set_holiday(data):
    us_hol_cal = calendar()
    holidays = us_hol_cal.holidays(start = data['pickup_datetime'].min(), end=data['pickup_datetime'].max())
    data['isWeekend'] = (data['pickup_datetime'].dt.dayofweek > 5).astype(int)
    data['isUSHoliday'] = (data['pickup_datetime'].isin(holidays)).astype(int)
    data['isHoliday'] = data['isWeekend'] | data['isUSHoliday']
    
data = readData('test_new.csv')
#    print(len(data))

data['dist'] = \
haversine_np(data.pickup_longitude, data.pickup_latitude,\
             data.dropoff_longitude, data.dropoff_latitude)
#add peak and holiday for better analysis
set_peak(data)
set_holiday(data)
#display(data.head(20))
#    print(len(data))
data = data.drop('isWeekend', axis=1)
data = data.drop('store_and_fwd_flag',axis=1)
data = data.drop('isUSHoliday',axis=1)
data.to_csv('testok.csv') #write csv
'''

infile = open('testok.csv','r')
data = infile.readlines()[1:]
all_list = []
sample_all = []
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

trunk1 = get_month_day_time(all_list)
#print(trunk1)
#print(trunk1.shape)
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
        col5[i][0] = float(data_list[i][5])
        col6[i][0] = float(data_list[i][6])
        col7[i][0] = float(data_list[i][7])
        col8[i][0]= float(data_list[i][8])
        col9[i][0] = int(data_list[i][11])
        col10[i][0] = int(data_list[i][12])
    return np.concatenate((col4,col5,col6,col7,col8,col9,col10),axis = 1)
trunk2 = get_vendor_lon1_lat1_lon2_lat2_isPeak_isHol(all_list)
#print(trunk2)
#print(trunk2.shape)
matrix = np.concatenate((trunk1,trunk2),axis = 1)

#print(len(matrix))

def normal(sample):
    if sample[4] < -75 or sample[4] > -73.92:
        sample[4] = mean[4]
    if sample[5] < 40.7 or sample[5] > 40.82:
        sample[5] = mean[5]
    if sample[6] < -75 or sample[6] > -73.92:
        sample[6] = mean[6]
    if sample[7] < 40.7 or sample[7] > 40.82:
        sample[7] = mean[7]
    return sample
        
matrix1 = np.empty([10000, 10]) 
#sample_ = matrix[0]  
#print(sample_)
for i in range(len(matrix)):
    sample_ = matrix[i]
#    print(sample_)
    sample = normal(sample_)
#    print(sample)
    matrix1[i] = sample

print(matrix1.shape)

nor_matrix = (matrix1-mean)/std
#sample = nor_matrix[len(matrix)-1]
#print(sample)
#print(mean)
#print(std)

#pre_list = np.zeros((len(nor_matrix),1),dtype = int)
#print(pre_list.shape)
#print(np.array([nor_matrix[0]]))
#print(np.array([nor_matrix[0]]).shape)

from keras.models import load_model
model = load_model('my_model_7.h5')
#pre = model.predict(np.array([nor_matrix[0]]))
#print(pre)
#print(pre[0])
#print(type(pre[0][0]))
#print(type(pre))
#print(pre.shape)

#pre = model.predict(np.array([sample]))
pre = model.predict_on_batch(nor_matrix)
#print(pre[0][0])

data1 = readData('testok.csv')
data1['trip_duration'] = int(pre)
data1.to_csv('testcomplete_model7.1.csv')



'''
data2 = readData('testcomplete_model4.csv')
print(len(data2))

data3 = readData('testcomplete_model7.csv')
cnt = 0
for i in range(len(data2)):
    if abs(data2['trip_duration'][i] - data3['trip_duration'][i]) > 300:
        cnt += 1
print(cnt)
#print(data2['trip_duration'] - data3['trip_duration'])
'''
