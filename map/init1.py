#Import Flask Library
from flask import Flask, render_template, request, session, url_for, redirect
from datetime import date
import holidays
import numpy as np
import pickle
from keras.models import load_model

#Initialize the app from Flask
app = Flask(__name__)
app.debug = True

@app.route("/", methods=['GET', 'POST'])
def hello():
	return render_template('index.html')

@app.route("/map", methods=['GET', 'POST'])
def map():
	longitude1 = request.form['longitude1']
	latitude1 = request.form['latitude1']
	longitude2 = request.form['longitude2']
	latitude2 = request.form['latitude2']
	vendor = request.form['vendor']
	vendor = int(vendor)
	date = request.form['datepicker']
	hour = request.form['hour']
	minute = request.form['minute']
	def isPeak(hour):
		if int(hour) in [7,8,9,18,19,20,21]:
			return 1
		else:
			return 0
	def isHoliday(date):
		us_holidays = holidays.UnitedStates()
		if date in us_holidays:
			return 1
		else:
			return 0
	ispeak = isPeak(hour)
	isholiday = isHoliday(date)


	
	infile = open('/Users/duanwujie/Desktop/Map-project/map/static/archive.dat','rb')
	dic = pickle.load(infile)
	mean = dic['mean']
	std = dic['std']

	temp = date.split('/')
	month = int(temp[0])
	day = int(temp[1])
	hour = int(hour)
	if hour >= 0 and hour < 6:
	    hour = 0
	elif hour >= 6 and hour < 12:
	    hour = 1
	elif hour >= 12 and hour < 18:
	    hour = 2
	elif hour >= 18 and hour < 24:
	    hour = 3
	lon1 = float(longitude1)
	lat1 = float(latitude1)
	lon2 = float(longitude2)
	lat2 = float(latitude2)

	
	sample = np.array([[month,day,hour,vendor,lon1,lat1,lon2,lat2,ispeak,isholiday]])
	#sample = np.array([[3,25,2,2,-73.98300171,40.7557106,-73.97595978,40.76294327,0,0]])
	#print(sample.shape)
	sample_nor = (sample-mean)/std
	#print(sample_nor)
	
	model = load_model('/Users/duanwujie/Desktop/Map-project/map/static/my_model_7.h5')
	pre = model.predict(sample_nor)
	pre = pre[0][0]
	prem = int(pre//60)
	pres = int(pre % 60)
	prem = str(prem)
	pres = str(pres)


	if (pre):
		return render_template('index.html',data = prem, data1 = pres, lon1 = longitude1, lat1 = latitude1, lon2 = longitude2, lat2 = latitude2)

#Run the app on localhost port 5000
#debug = True -> you don't have to restart flask
#for changes to go through, TURN OFF FOR PRODUCTION
app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'

if __name__ == "__main__":
    app.run()