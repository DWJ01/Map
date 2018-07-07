#Import Flask Library
from flask import Flask, render_template, request, session, url_for, redirect
from datetime import date
import holidays

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
	date = request.form['date']
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
	if (longitude1 and latitude1):
		return render_template('index.html',data = longitude1, ispeak = ispeak, isholiday = isholiday)

#Run the app on localhost port 5000
#debug = True -> you don't have to restart flask
#for changes to go through, TURN OFF FOR PRODUCTION
app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'

if __name__ == "__main__":
    app.run()