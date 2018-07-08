#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 23:23:58 2018

@author: liuconnie
"""

#This is a file conducting data cleaning process

import pandas as pd 
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

earth_radius = 6371

#This is a function to access data in a csv file
def readData(filename):
    data = pd.read_csv(filename, parse_dates=[2,3])
    return data

#This is a function calculating great circle distance between to points 
#with given longitude and latitude in numpy array
def haversine_np(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat):

    pickup_lon, pickup_lat, dropoff_lon, dropoff_lat = map(np.radians, \
                [pickup_lon, pickup_lat, dropoff_lon, dropoff_lat])

    dlon = dropoff_lon-pickup_lon
    dlat = dropoff_lat-pickup_lat

    r = np.sin(dlat/2.0)**2+np.cos(pickup_lat)*np.cos(dropoff_lat)*np.sin(dlon/2.0)**2

    circle = 2*np.arcsin(np.sqrt(r))
    return 6367*circle

#This is a function to check whether the cab order is produced in a rush hour
def set_peak(data):
    data.set_index('pickup_datetime', inplace=True, drop=False, append=False)
    data['pickup_hour'] = data.index.hour
    data['isPeak'] = (data['pickup_hour'].isin([7,8,9,18,19,20,21])).astype(int)
    #We assume that rush hour in NY is 7:00-9:00 and 18:00-21:00
  
#This is a function to check whether the cab order is produced in holidays
#We define both weekends and USholidays are holidays
def set_holiday(data):
    us_holi = calendar()
    holidays = us_holi.holidays(start = data['pickup_datetime'].min(), end=data['pickup_datetime'].max())
    data['isWeekend'] = (data['pickup_datetime'].dt.dayofweek > 5).astype(int)
    data['isUSHoliday'] = (data['pickup_datetime'].isin(holidays)).astype(int)
    data['isHoliday'] = data['isWeekend'] | data['isUSHoliday']
    
    
def main():
    data = readData('train.csv') #access data
    #The original data size
    print(len(data)) #1312780

    """===========datacleaning process aiming at outliers==============="""
    #clean order carrying 0 passengers
    data=data[data.passenger_count!=0]

    #Below we first display boxplot of some variables then clean outliers
    """-----1.trip duration-----"""
    #data.boxplot(column=['trip_duration'])
    data = data[data.trip_duration < 20000]
    #data.boxplot(column=['trip_duration']);
    data = data[data.trip_duration > 200];
    #data.boxplot(column=['trip_duration'])
    #We finally select trip_duration between(200,20000) as normal duration
    
    
    """-----2.pickup & dropoff location-----"""
    #data.boxplot(column=['pickup_longitude', 'dropoff_longitude'])
    #data.boxplot(column=['pickup_longitude', 'dropoff_longitude'])
    pickup_LL = data['pickup_longitude'] > -75
    pickup_UL = data['pickup_longitude'] < -73.92
    dropoff_LL = data['dropoff_longitude'] > -75
    dropoff_UL = data['dropoff_longitude'] < -73.92
    data = data[pickup_LL & pickup_UL & dropoff_LL & dropoff_UL]
    #data.boxplot(column=['pickup_longitude', 'dropoff_longitude'])
    #We finally select longitude between(-75,-73.92)

    pickup_LL = data['pickup_latitude'] > 40.7
    pickup_UL = data['pickup_latitude'] < 40.82
    dropoff_LL = data['dropoff_latitude'] > 40.7
    dropoff_UL = data['dropoff_latitude'] < 40.82
    data = data[pickup_LL & pickup_UL & dropoff_LL & dropoff_UL]
    #data.boxplot(column=['pickup_longitude', 'dropoff_longitude'])
    #We finally select latitude between(40.7, 40.82)
    

    """-----3.distance & speed-----"""
    #call a function to calculate big circle distance (in km)
    data['dist'] = \
    haversine_np(data.pickup_longitude, data.pickup_latitude,\
                 data.dropoff_longitude, data.dropoff_latitude) 
    #calculate approximate speed (in km/h) 
    data['speed'] = data['dist']*3600/data['trip_duration']
    #data.boxplot(column=['dist', 'speed'])
    dist_min = data['dist']>0.1
    speed_min = data['speed']>0.0
    speed_max = data['speed']<250.0
    #We finally select distance more than 0.1km
    #We finally select speed between (0.0,250.0)km/h
    data = data[dist_min & speed_min & speed_max]
    

    "===========add peak and holiday for better analysis==========="
    set_peak(data)
    set_holiday(data)

    #The data size after cleaning
    print(len(data)) #1044036
    
    #Clear util columns
    data = data.drop('isWeekend', axis=1)
    data = data.drop('store_and_fwd_flag',axis=1)
    data = data.drop('isUSHoliday',axis=1)

    data.to_csv('finaldropped.csv') #export csv
main()
        
    