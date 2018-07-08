#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 10:36:31 2018

@author: liuconnie
"""


import pandas as pd
import matplotlib.pyplot as plt

#This is a plot of all pickup and dropoff location
#which is a preprocessing step to see density of cab pickup and dropoff  
df = pd.read_csv('train.csv')
xlim = [-74.03, -73.77]
ylim = [40.63, 40.85]

df = df[(df.pickup_longitude> xlim[0]) & (df.pickup_longitude < xlim[1])]
df = df[(df.dropoff_longitude> xlim[0]) & (df.dropoff_longitude < xlim[1])]
df = df[(df.pickup_latitude> ylim[0]) & (df.pickup_latitude < ylim[1])]
df = df[(df.dropoff_latitude> ylim[0]) & (df.dropoff_latitude < ylim[1])]

#longitude = list(df.pickup_longitude)
#latitude = list(df.pickup_latitude)

longitude = list(df.dropoff_longitude)
latitude = list(df.dropoff_latitude)
plt.figure(figsize = (10,10))
plt.plot(longitude,latitude,'.', alpha = 0.6, markersize = 0.05,color='purple')

plt.savefig("dropoff.png")
plt.show()