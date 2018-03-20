# To run:
# - Install necessary dependencies
# - For now, only BTC is supported (because of API limitations)
# - Make desired changes to predict_price() call at the bottom
#	(Smaller time intervals will take longer to compute)
# - Run as Python script

from __future__ import division
import numpy as np
import pandas
import requests
import datetime
from datetime import timedelta
from time import sleep
from twitterscraper import query_tweets

########################################################################

# Global constants
CRYPTO_START_DAY = datetime.datetime(2018, 1, 1)

SECONDS_IN_MINUTE = 60
SECONDS_IN_FIVE_MINUTES = 300
SECONDS_IN_FIFTEEN_MINUTES = 900
SECONDS_IN_HOUR = 3600
SECONDS_IN_SIX_HOURS = 21600
SECONDS_IN_DAY = 86400


########################################################################

class Coinmap(object):
	"""For fetching businesses that accept cryptocurrency
	"""
	def __init__(self):
		self.uri = 'https://coinmap.org/api/v1/venues/'

	def get_venues_between(self, after, before):
		"""Get venues which began accepting crypto between dates.

		Args:
			after (datetime): Date after which to start searching.
			before (datetime): Date before which to stop searching.
		Returns:
			(int): Number of veunes which opened in this window.
		"""
		result = requests.get(self.uri, {
			'after': Coinmap.__convert_date(after),
			'before': Coinmap.__convert_date(before)
			})
		venues = result.json()['venues']
		return len(venues)

	def get_venues_after(self, after):
		return self.get_venues_between(after, datetime.datetime.now())

	@staticmethod
	def __convert_date(date):
		return '{year}-{month:02d}-{day:02d}'.format(
			year = date.year,
			month = date.month,
			day = date.day)


class GDAX(object):
	"""Currency fetching
	Help from: https://hackercrypt.com/code/fetch-historic-price-data

	"""
	MAX_RETRIES = 3

	def __init__(self, pair):
		"""Create the exchange object.

	    Args:
	      pair (str): Examples: 'BTC-USD', 'ETH-USD'...
	    """
		self.pair = pair
		self.uri = 'https://api.gdax.com/products/{pair}/candles'.format(pair=self.pair)

	def fetch(self, start, end, granularity):
		"""Fetch the candle data for a given range and granularity.
		
		Args:
			start (datetime): The start of the date range.
			end (datetime): The end of the date range (excluded).
			granularity (int): The granularity of the candles data (in minutes).
		Returns:
			(pandas.DataFrame): A data frame of the OHLC and volume information, indexed by their unix timestamp.
		"""	
		print "Attempting to connect to GDAX..."

		data = []
		# Get maximum of 200 data points at a time
		delta = timedelta(seconds=granularity * 200)

		slice_start = start
		while slice_start != end:
			slice_end = min(slice_start + delta, end)
			data += self.request_slice(slice_start, slice_end, granularity)
			slice_start = slice_end

		data_frame = pandas.DataFrame(data=data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
		data_frame.set_index('time', inplace=True)
		return data_frame

	def request_slice(self, start, end, granularity):
		for retry_count in xrange(0, self.MAX_RETRIES):
			response = requests.get(self.uri, {
				'start': GDAX.__date_to_iso8601(start),
				'end': GDAX.__date_to_iso8601(end),
				'granularity': granularity
				})

			if response.status_code != 200 or not len(response.json()):
				if retry_count + 1 == self.MAX_RETRIES:
					raise Exception('Failed to get exchange data for ({}, {})'.format(start, end))
				else:
					# Back off exponentially
					sleep(1.5 ** retry_count)
			else:
				# Sort historic rates
				result = sorted(response.json(), key=lambda x: x[0])
				return result

	@staticmethod
	def __date_to_iso8601(date):
		"""Convert a datetime object to the ISO-8601 date format (expected by the GDAX API).
		
		Args:
			date (datetime): The date to be converted.
		Returns:
			str: The ISO-8601 formatted date.
		"""
		return '{year}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}'.format(
			year = date.year,
			month = date.month,
			day = date.day,
			hour = date.hour,
			minute = date.minute,
			second = date.second)

########################################################################



########################################################################

def get_historic_constant(time_and_price_matrix):
	"""Gets historic constant values from a time and price matrix

	Args:
		time_and_price_matrix (pandas.DataFrame): Matrix with just time and crypto price
	Returns:
		(numpy.array): Prices from one time period ago.
	"""
	print "Researching how the cryptocurrency has performed in the past..."
	X = time_and_price_matrix.as_matrix().flatten()
	X = X[1: ]
	print "Past trends analyzed."
	print
	return np.append(X, 0)

def get_base_constant(time_and_price_matrix):
	"""Gets the base constant for how widely the cryptocurrency can be used

	Args:
		time_and_price_matrix (pandas.DataFrame): Matrix with just time and crypto price
	Returns:
		(numpy.array): How many companies accepted crypto at a give time period
	"""
	print "Searching for how many businesses are taking the currency..."
	arr = time_and_price_matrix.index.values
	to_datetime = np.vectorize(datetime.datetime.fromtimestamp)
	datetime_list = to_datetime(arr[0: ])

	cmap = Coinmap()
	get_businesses = np.vectorize(cmap.get_venues_after)
	num_businesses = get_businesses(datetime_list)
	print "Historic business data analyzed."
	print
	return num_businesses

def get_press_constant(time_and_price_matrix):
	print "Seeing how the press discusses the cryptocurrency..."
	print "Press data analyzed."
	print
	return np.zeros(time_and_price_matrix.shape[0])

def get_social_constant(time_and_price_matrix, interval):
	"""Gets the social constant for how much the cryptocurrency is discussed on Twitter

	Args:
		time_and_price_matrix (pandas.DataFrame): Matrix with just time and crypto price
		interval (int): Number of seconds between queries
	Returns:
		(numpy.array): How many tweets were made about the cryptocurrency over a given time
	"""
	print "Analyzing cryptocurrency's social standing..."
	arr = time_and_price_matrix.index.values
	to_datetime = np.vectorize(datetime.datetime.fromtimestamp)
	datetime_list = to_datetime(arr[0: ])

	delta = timedelta(2)
	get_tweets = lambda d: len(query_tweets("Bitcoin OR BTC", limit=50, begindate=d.date() - delta, enddate=d.date()))
	get_tweets = np.vectorize(get_tweets)

	num_tweets = get_tweets(datetime_list)
	print "Social standing analyzed."
	print
	return num_tweets

def get_time_and_price_change(crypto_data):
	"""Extracts time and price change vectors out of historical crypto data.

	Args:
		crypto_data (pandas.DataFrame): Historical data.
	Returns:
		(pandas.DataFrame): Data containing only the timestamp and change over the interval
	"""
	data = crypto_data.filter(['time'], axis=1)
	data['price_change'] = crypto_data['close'] - crypto_data['open']
	return data

def get_current_price(price_data):
	"""Gets the current price from a data frame of historic data.

	Args:
		price_data (pandas.DataFrame): Data to get the current price from.
	Returns:
		(float): The closing price from the last available data point.
	"""
	return price_data['close'].tail(1).values[0]

def get_historic_data(crypto, interval):
	"""Gets the historic data from a cryptocurrency

	Args:
		crypto (str): The crypto currency to fetch data for
		interval (int): Seconds between data points to fetch
	Returns:
		(pandas.DataFrame): historic data for the currency
	"""
	print "Fetching historic data..."
	today = datetime.datetime.now()
	start = CRYPTO_START_DAY

	data = GDAX('{crypto}-USD'.format(crypto=crypto)).fetch(start, today, interval)

	print "Received historic data."
	print
	return data

########################################################################

def predict_price(crypto, time_from_now):
	"""Predicts the price of a crypto currency

	Args:
		crypto (str): The cypto currency to predict.
		time_from_now (int): Number of seconds into the future to predict
	Returns:
		(float, float): Tuple containing predicted price and confidence
	"""
	print
	print "SimpleCoin: Making Investment Simple"
	print
	print "This algorithm will predict a cryptocurrency's value at any time in the future."
	print "On this round, we'll predict what {crypto} will be trading at {time_from_now} seconds from now.".format(crypto=crypto, time_from_now=time_from_now)
	print

	interval = time_from_now

	historic_data = get_historic_data(crypto, interval)
	curr_price = get_current_price(historic_data)

	prediction_matrix = get_time_and_price_change(historic_data)

	S = get_social_constant(prediction_matrix, interval)
	P = get_press_constant(prediction_matrix)
	B = get_base_constant(prediction_matrix)
	X = get_historic_constant(prediction_matrix)

	print "And finally, some linear algebra to finish the algorithm"
	print 
	# A = constants for each time interval
	# b = price change values at each time interval
	# Ax* ~ b <- least squares approximate solution
	A = np.column_stack((S, P, B, X))
	b = prediction_matrix.as_matrix()
	x_star = np.linalg.lstsq(A, b)[0].flatten()

	# Constant vector = [S, P, B, X]
	# Delta_P = S*Cs + P*Cp + B*Cb + X*Cx
	constant_vector = A[0]
	delta_price = np.dot(x_star, constant_vector)
	price = curr_price + delta_price

	# Confidence = Ax* [dot] b
	approx = np.dot(A, x_star)
	confidence = np.inner(approx, b.flatten()) / (np.linalg.norm(approx) * np.linalg.norm(b.flatten()))

	return price, confidence


########################################################################

price, confidence = predict_price('BTC', SECONDS_IN_DAY)
print 'We are {confidence} confident that the price will be {price}'.format(confidence=confidence, price=price)