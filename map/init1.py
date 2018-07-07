#Import Flask Library
from flask import Flask, render_template, request, session, url_for, redirect

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
	if (longitude1 and latitude1):
		return render_template('index.html',data = longitude1)

#Run the app on localhost port 5000
#debug = True -> you don't have to restart flask
#for changes to go through, TURN OFF FOR PRODUCTION
app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'

if __name__ == "__main__":
    app.run()