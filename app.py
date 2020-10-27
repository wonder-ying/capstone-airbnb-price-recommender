import logging, io, os, sys
import flask
from flask import Flask, request, jsonify, render_template
import pickle
import bz2
import _pickle as cPickle
import json
import pandas as pd
import numpy as np

app = Flask(__name__)

rf_model = None
column_name = [
	 'borough', 
	 'instant_bookable',
	 'room_type',
	 'property_type',
	 'host_is_superhost',
	 'host_total_listings_count',
	 'accommodates',
	 'bedrooms',
	 'bathrooms',
	 'minimum_nights',
	 'maximum_nights',
	 'security_deposit',
	 'cleaning_fee',
	 'coffee_machine',
	 'outdoor_space']

df = pd.read_pickle('static/data_nn.pkl')
# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data

@app.before_first_request
def startup():
    global rf_model 
    global nn_model
    # rf model
    rf_model = decompress_pickle('static/rf_model.pbz2')
    # nn model
    with open('static/similar_listing.pickle', 'rb') as f:
        nn_model = pickle.load(f)


@app.errorhandler(500)
def server_error(e):
    logging.exception('some eror')
    return """
    And internal error <pre>{}</pre>
    """.format(e), 500

@app.route("/", methods=['POST', 'GET'])
def index():
	return flask.render_template('home.html')

# Pass from data to prediction model
def predict_price(inputs):
    data = pd.DataFrame([inputs], columns=column_name)
    return round(rf_model.predict(data)[0], 2)

def predict_listing(inputs):
	data = pd.DataFrame([inputs], columns=['price'] + column_name)
	recs = nn_model['knn'].kneighbors(nn_model['preprocessor'].transform(data), return_distance = False)
	rec_df = pd.DataFrame(columns = df.columns.tolist())
	for i in recs:
		rec_df = rec_df.append(df.iloc[i], ignore_index=True)

	return rec_df[['listing_url', 'borough', 'price']]



@app.route('/result', methods = ['POST'])
def result():
	if request.method == 'POST':
		borough = request.form['borough']
		instant_bookable = int(request.form['instant_bookable'])
		room_type = request.form['room_type']
		property_type = request.form['property_type']
		host_is_superhost = int(request.form['host_is_superhost'])
		host_total_listings_count = int(request.form['host_total_listings_count'])
		accommodates = int(request.form['accommodates'])
		bedrooms = int(request.form['bedrooms'])
		bathrooms = int(request.form['bathrooms'])
		minimum_nights = int(request.form['minimum_nights'])
		maximum_nights = int(request.form['maximum_nights'])
		security_deposit = float(request.form['security_deposit'])
		cleaning_fee = float(request.form['cleaning_fee'])
		coffee_machine = int(request.form['coffee_machine'])
		outdoor_space = int(request.form['outdoor_space'])
		inputs = [borough, instant_bookable, room_type, property_type, host_is_superhost, host_total_listings_count, accommodates, bedrooms, bathrooms, minimum_nights, maximum_nights, security_deposit, cleaning_fee, coffee_machine, outdoor_space]
		result = predict_price(inputs)
		nn_inputs = [result] + inputs
		sim_list = predict_listing(nn_inputs)
		sim_list.columns = ['Links', 'Neighborhood', 'Price']

		return render_template("home.html", table=sim_list.to_html(index=False, render_links=True), prediction = result, borough = borough, instant_bookable = instant_bookable, room_type = room_type, property_type = property_type, host_is_superhost = host_is_superhost, host_total_listings_count = host_total_listings_count, accommodates = accommodates, bedrooms = bedrooms, bathrooms = bathrooms, minimum_nights = minimum_nights, maximum_nights = maximum_nights, security_deposit = security_deposit, cleaning_fee = cleaning_fee, coffee_machine = coffee_machine, outdoor_space = outdoor_space)
	# response = jsonify({
	# 	# populate the estimated price to the frontend
	# 	'estimated_price': util.predict_price(inputs)
	# 	})
	# reponse.headers.add('Access-Control-Allow-Origin', '*')

	#return response
if __name__ == '__main__':
    app.run(port=8000, debug=True)
