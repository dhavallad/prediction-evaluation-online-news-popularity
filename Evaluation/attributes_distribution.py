# Title		 	: attributes_distribution.py
# Description 	: This file generate the distribtuion graph for most of all the attribtues.
# Author 		: Dhaval Lad
# Contact 		: dhavallad92@gmail.com

__author__ = 'Dhaval Lad'

import os,sys
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from time import time
import xlsxwriter
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.graph_objs import *
# import matplotlib.pyplot as plt; plt.rcdefaults()



def channel_distribution(file):
	"""
	This Generate the distribution graph for data in input file. Particualrly in this 
	function it generate for different news category.
	Args:
		fname : Input file name. e.g. OnlineNewsPopularity.csv
	"""
	channel_list = [ 'Lifestyle', 'Entertainment', 'Business', 'Social Media', 'Techology', 'World']
	df=pd.read_csv(file)  # load dataset to padas dataframe
	channel_dic = {}
	# print channel_dic
	col_channel =  list(df.columns[13:19])
	for channel,c in zip(channel_list,col_channel):
		channel_dic[channel] = df[c].value_counts()[1]
	data = [go.Bar(x= channel_dic.keys(),y=channel_dic.values())]
	layout = dict(title = 'Distribution of Channel',
              xaxis = dict(title = 'Channel'),
              yaxis = dict(title = 'Frequency'),
              )
	fig = go.Figure(data=data, layout=layout)
	plot_url = py.plot(fig, filename='channel-distribution', auto_open=False)
	print "Success"



def weekdays_distribution(file):
	"""
	This Generate the distribution graph for data in input file. Particualrly in this 
	function it generate for different weekdays when aticle is published.
	Args:
		fname : Input file name. e.g. OnlineNewsPopularity.csv
	"""

	# csv_filename="OnlineNewsPopularity.csv"
	df=pd.read_csv(file)  # load dataset to padas dataframe
	weekday_dic = {}
	col_days=list(df.columns[31:38])
	# weekday_list = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
	# print col_days
	# ab = df['weekday_is_wednesday'].value_counts()[1]
	for i in col_days:
		weekday_dic[i[11:]] = df[i].value_counts()[1]
	# print df['weekday_is_wednesay'].value_counts()
	# print weekday_dic
	# print weekday_dic.values()
	data = [go.Bar(x= weekday_dic.keys(),y=weekday_dic.values())]
	layout = dict(title = 'Distribution of Weekdays',
              xaxis = dict(title = 'Weekdays'),
              yaxis = dict(title = 'Frequency'),)
	fig = go.Figure(data=data, layout=layout)
	plot_url = py.plot(fig, filename='weekday-distribution', auto_open=False)
	print "Sucess"


def n_tokens_title_distribution(file):
	"""
	This Generate the distribution graph for data in input file. Particualrly in this 
	function it generate for different n_tokens_title for articles.
	Args:
		fname : Input file name. e.g. OnlineNewsPopularity.csv
	"""

	# csv_filename="OnlineNewsPopularity.csv"
	df=pd.read_csv(file)  # load dataset to padas dataframe
	weekday_dic = {}
	
	data = [go.Scatter(x= df['n_tokens_title'],y=df['shares'],mode = 'markers')]
	layout = dict(title = 'Distribution of n_tokens_title',
              xaxis = dict(title = 'n_tokens_title'),
              yaxis = dict(title = 'Shares'),)
	fig = go.Figure(data=data, layout=layout)
	plot_url = py.plot(fig, filename='n_tokens_title-distribution', auto_open=False)
	print "Sucess"

def n_tokens_content_distribution(file):
	"""
	This Generate the distribution graph for data in input file. Particualrly in this 
	function it generate for different n_tokens_content for articles.
	Args:
		fname : Input file name. e.g. OnlineNewsPopularity.csv
	"""

	# csv_filename="OnlineNewsPopularity.csv"
	df=pd.read_csv(file)  # load dataset to padas dataframe
	weekday_dic = {}
	
	data = [go.Scatter(x= df['n_tokens_content'],y=df['shares'],mode = 'markers')]
	layout = dict(title = 'Distribution of n_tokens_content',
              xaxis = dict(title = 'n_tokens_content'),
              yaxis = dict(title = 'Shares'),)
	fig = go.Figure(data=data, layout=layout)
	plot_url = py.plot(fig, filename='n_tokens_content-distribution', auto_open=False)
	print "Sucess"

def n_unique_tokens_distribution(file):
	"""
	This Generate the distribution graph for data in input file. Particualrly in this 
	function it generate for different n_unique_tokens for articles.
	Args:
		fname : Input file name. e.g. OnlineNewsPopularity.csv
	"""

	# csv_filename="OnlineNewsPopularity.csv"
	df=pd.read_csv(file)  # load dataset to padas dataframe
	weekday_dic = {}
	
	data = [go.Scatter(x= df['n_unique_tokens'],y=df['shares'],mode = 'markers')]
	layout = dict(title = 'Distribution of n_unique_tokens',
              xaxis = dict(title = 'n_unique_tokens'),
              yaxis = dict(title = 'Shares'),)
	fig = go.Figure(data=data, layout=layout)
	plot_url = py.plot(fig, filename='n_unique_tokens-distribution', auto_open=False)
	print "Sucess"


def kw_avg_max_distribution(file):
	"""
	This Generate the distribution graph for data in input file. Particualrly in this 
	function it generate for different kw_avg_max for articles.
	Args:
		fname : Input file name. e.g. OnlineNewsPopularity.csv
	"""

	# csv_filename="OnlineNewsPopularity.csv"
	df=pd.read_csv(file)  # load dataset to padas dataframe
	weekday_dic = {}
	
	data = [go.Scatter(x= df['kw_avg_max'],y=df['shares'],mode = 'markers')]
	layout = dict(title = 'Distribution of kw_avg_max',
              xaxis = dict(title = 'kw_avg_max'),
              yaxis = dict(title = 'Shares'),)
	fig = go.Figure(data=data, layout=layout)
	plot_url = py.plot(fig, filename='kw_avg_max-distribution', auto_open=False)
	print "Sucess"


def global_subjectivity_distribution(file):
	"""
	This Generate the distribution graph for data in input file. Particualrly in this 
	function it generate for different global_subjectivity for articles.
	Args:
		fname : Input file name. e.g. OnlineNewsPopularity.csv
	"""

	# csv_filename="OnlineNewsPopularity.csv"
	df=pd.read_csv(file)  # load dataset to padas dataframe
	weekday_dic = {}
	
	data = [go.Scatter(x= df['global_subjectivity'],y=df['shares'],mode = 'markers')]
	layout = dict(title = 'Distribution of global_subjectivity',
              xaxis = dict(title = 'global_subjectivity'),
              yaxis = dict(title = 'Shares'),)
	fig = go.Figure(data=data, layout=layout)
	plot_url = py.plot(fig, filename='global_subjectivity-distribution', auto_open=False)
	print "Sucess"


def average_token_length_distribution(file):
	"""
	This Generate the distribution graph for data in input file. Particualrly in this 
	function it generate for different average_token_length for articles.
	Args:
		fname : Input file name. e.g. OnlineNewsPopularity.csv
	"""

	# csv_filename="OnlineNewsPopularity.csv"
	df=pd.read_csv(file)  # load dataset to padas dataframe
	weekday_dic = {}
	
	data = [go.Scatter(x= df['average_token_length'],y=df['shares'],mode = 'markers')]
	layout = dict(title = 'Distribution of average_token_length',
              xaxis = dict(title = 'average_token_length'),
              yaxis = dict(title = 'Shares'),)
	fig = go.Figure(data=data, layout=layout)
	plot_url = py.plot(fig, filename='average_token_length-distribution', auto_open=False)
	print "Sucess"


def title_sentiment_polarity_distribution(file):
	"""
	This Generate the distribution graph for data in input file. Particualrly in this 
	function it generate for different title_sentiment_polarity for articles.
	Args:
		fname : Input file name. e.g. OnlineNewsPopularity.csv
	"""

	# csv_filename="OnlineNewsPopularity.csv"
	df=pd.read_csv(file)  # load dataset to padas dataframe
	weekday_dic = {}
	
	data = [go.Scatter(x= df['title_sentiment_polarity'],y=df['shares'],mode = 'markers')]
	layout = dict(title = 'Distribution of title_sentiment_polarity',
              xaxis = dict(title = 'title_sentiment_polarity'),
              yaxis = dict(title = 'Shares'),)
	fig = go.Figure(data=data, layout=layout)
	plot_url = py.plot(fig, filename='title_sentiment_polarity-distribution', auto_open=False)
	print "Sucess"

def LDA_00_distribution(file):
	"""
	This Generate the distribution graph for data in input file. Particualrly in this 
	function it generate for different LDA_00 for articles.
	Args:
		fname : Input file name. e.g. OnlineNewsPopularity.csv
	"""

	# csv_filename="OnlineNewsPopularity.csv"
	df=pd.read_csv(file)  # load dataset to padas dataframe
	weekday_dic = {}
	
	data = [go.Scatter(x= df['LDA_00'],y=df['shares'],mode = 'markers')]
	layout = dict(title = 'Distribution of LDA_00',
              xaxis = dict(title = 'LDA_00'),
              yaxis = dict(title = 'Shares'),)
	fig = go.Figure(data=data, layout=layout)
	plot_url = py.plot(fig, filename='LDA_00-distribution', auto_open=False)
	print "Sucess"




def run(filename):
	print filename
	channel_distribution(filename)
	weekdays_distribution(filename)
	n_tokens_title_distribution(filename)
	n_tokens_content_distribution(filename)
	n_unique_tokens_distribution(filename)
	kw_avg_max_distribution(filename)
	global_subjectivity_distribution(filename)
	average_token_length_distribution(filename)
	title_sentiment_polarity_distribution(filename)
	LDA_00_distribution(filename)


if __name__ == '__main__':
	run(sys.argv[1])