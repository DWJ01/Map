#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 23:23:58 2018

@author: liuconnie
"""
"""from math import sin, asin, cos, radians, fabs, sqrt
earth_radius = 6371
def hav(theta):
    s = sin(theta/2)
    return s*s
def get_dist_hav(lat0,lon0,lat1,lon1):
    lat0 = radians(lat0)
    lon0 = radians(lon0)
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    
    dlon = fabs(lon0 - lon1)
    dlat = fabs(lat0 - lat1)
    h = hav(dlat)+cos(lat0)*cos(lat1)*hav(dlon)
    dist = 2*earth_radius*asin(sqrt(h))
    return dist"""
 
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from math import floor # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

import pandas as pd#for dataprocessing.
import matplotlib.pyplot as plt
from IPython.display import display
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from math import sin, asin, cos, radians, fabs, sqrt
earth_radius = 6371
#matplotlib inline

def readData(filename):
    data = pd.read_csv(filename, parse_dates=[2,3])
    #display(data.head(3))
    return data

"""def hav(theta):
    s = sin(theta/2)
    return s*s"""

"""def get_dist_hav(lat0,lon0,lat1,lon1):
    lat0 = radians(lat0)
    lon0 = radians(lon0)
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    
    dlon = fabs(lon0 - lon1)
    dlat = fabs(lat0 - lat1)
    h = hav(dlat)+cos(lat0)*cos(lat1)*hav(dlon)
    dist = 2*earth_radius*asin(sqrt(h))
    return dist"""
    
"""def get_dist_hav(lat0,lon0,lat1,lon1):
    lat0 = radians(lat0)
    lon0 = radians(lon0)
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    
    dlon = fabs(lon0 - lon1)
    dlat = fabs(lat0 - lat1)
    h = hav(dlat)+cos(lat0)*cos(lat1)*hav(dlon)
    dist = 2*earth_radius*asin(sqrt(h))
    return dist"""

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
    
    
def main():
    data = readData('train.csv')
    print(len(data))
    """data['distance(km)'] = None
    data['speed(km/h)'] = None
    for i in range(len(data)):
        data['distance(km)'][i] = get_dist_hav(data['pickup_latitude'][i],data['pickup_longitude'][i],data['dropoff_latitude'][i],data['dropoff_longitude'][i])
        data['speed(km/h)'][i] = data['distance(km)'][i]*3600/data['trip_duration'][i]
    dist_min = data['distance(km)']>0.1
    speed_min = data['speed(m/s)']>0.0
    speed_max = data['speed(m/s)']<250.0
    data = data[dist_min & speed_min & speed_max]"""

    #clean 0 passengers
    data=data[data.passenger_count!=0]

    #clean outliners based on boxplot
    
    #display(data.isnull().sum())
    #data.boxplot(column=['trip_duration'])
    data = data[data.trip_duration < 20000]
    #data.boxplot(column=['trip_duration']);
    data = data[data.trip_duration > 200];
    #data.boxplot(column=['trip_duration'])
    #data.boxplot(column=['pickup_longitude', 'dropoff_longitude'])
    #data = data[ data.pickup_longitude > -80]
    #data.boxplot(column=['pickup_longitude', 'dropoff_longitude'])

    PICKUP_LOWER_LIMIT = data['pickup_longitude'] > -75
    PICKUP_UPPER_LIMIT = data['pickup_longitude'] < -73.92
    DROP_OFF_LOWER_LIMIT = data['dropoff_longitude'] > -75
    DROP_OFF_UPPER_LIMIT = data['dropoff_longitude'] < -73.92
    data = data[PICKUP_LOWER_LIMIT & PICKUP_UPPER_LIMIT & DROP_OFF_LOWER_LIMIT & DROP_OFF_UPPER_LIMIT]
    #pd.DataFrame.boxplot(data, column=['pickup_longitude', 'dropoff_longitude']);
    data.boxplot(column=['pickup_latitude', 'dropoff_latitude']);
    PICKUP_LOWER_LIMIT = data['pickup_latitude'] > 40.7
    PICKUP_UPPER_LIMIT = data['pickup_latitude'] < 40.82
    DROP_OFF_LOWER_LIMIT = data['dropoff_latitude'] > 40.7
    DROP_OFF_UPPER_LIMIT = data['dropoff_latitude'] < 40.82
    data = data[PICKUP_LOWER_LIMIT & PICKUP_UPPER_LIMIT & DROP_OFF_LOWER_LIMIT & DROP_OFF_UPPER_LIMIT]
    #pd.DataFrame.boxplot(data, column=['pickup_latitude', 'dropoff_latitude']);
    
    #display(data.head(20))
    #data['distance'] = data[['pickup_longitude'],['pickup_latitude'],['dropoff_longitude'],['dropoff_latitude']]\
    # .apply(get_dist_hav)
    data['dist'] = \
    haversine_np(data.pickup_longitude, data.pickup_latitude,\
                 data.dropoff_longitude, data.dropoff_latitude)
    data['speed'] = data['dist']*3600/data['trip_duration']
    dist_min = data['dist']>0.1
    speed_min = data['speed']>0.0
    speed_max = data['speed']<250.0

    data = data[dist_min & speed_min & speed_max]
    #data['distance'] = data.apply(get_dist_hav)
    #display(data.head(20))
    #add peak and holiday for better analysis
    set_peak(data)
    set_holiday(data)
    #display(data.head(20))
    print(len(data))
    data = data.drop('isWeekend', axis=1)
    data = data.drop('store_and_fwd_flag',axis=1)
    data = data.drop('isUSHoliday',axis=1)

    data.to_csv('finaldropped.csv') #write csv
main()
        
    