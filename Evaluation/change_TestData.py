# Title		 	: change_TestData.py
# Description 	: This file generating acurrancy and timing graph by changing in Testing Dataset.
# Author 		: Dhaval Lad
# Contact 		: dhavallad92@gmail.com

__author__ = 'Dhaval Lad'


import os,sys,math,json
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from time import time
import plotly.plotly as py
import plotly.graph_objs as go
from collections import OrderedDict
import matplotlib.pyplot as plt; plt.rcdefaults()

def percentile(data, percentile):
    """
    This function calcaute given percentile for given data.
    :return:
    """
    size = len(data)
    return sorted(data)[int(math.ceil((size * percentile) / 100)) - 1]

def changeTrain(file):
	"""
	This compare different alogrithm for data in input file. 
	Args:
		file : input file name. e.g. OnlineNewsPopularity.csv
	"""
	# Load dataset to padas dataframe
	# csv_filename="OnlineNewsPopularity.csv"
	df=pd.read_csv(file)
	perc = percentile(df['shares'], 50)
	# print perc
	popular = df.shares >= perc
	unpopular = df.shares < perc
	df.loc[popular,'shares'] = 1
	df.loc[unpopular,'shares'] = 0

	features=list(df.columns[2:60])
	# # split original dataset into 60% training and 40% testing
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(df[features], df['shares'], test_size=0.4, random_state=0)

	dataset_perc = []
	graph_dt = []
	graph_rndm = []
	graph_knn = []
	graph_nb = []
	graph_lr = []

	time_knn =[]
	time_dt = []
	time_rndm = []
	time_nb = []
	time_lr = []

	# Comparison of all alorithm and measuure accurancy and time.
	# Increasingly add size of training set 5% of orginal, keep testing size unchanged
	for i in range(0,100,5):
		X_rest, X_Test_Change, y_rest, y_Test_Change = cross_validation.train_test_split(X_test, y_test, test_size=0.049+i/100.0, random_state=0)
		print "------ Loop for training size is %s perc of original data --------- " %i 

		# Calculation for Decision Tree
		t_start=time()
		print "DecisionTree"
		dt = DecisionTreeClassifier(min_samples_split=20,random_state=99)
		clf_dt=dt.fit(X_train,y_train)
		score_dt=clf_dt.score(X_Test_Change,y_Test_Change)
		t_end=time()
		t_dt = t_end - t_start
		print "\n"

		# Calculation for KNN
		t_start=time()
		print "KNN"
		knn = KNeighborsClassifier()
		clf_knn=knn.fit(X_train, y_train)
		score_knn=clf_knn.score(X_Test_Change,y_Test_Change) 
		t_end=time()
		t_knn=t_end - t_start
		print "\n"

		# Calculation for Random Forest
		t_start=time()
		print "RandomForest"
		rf = RandomForestClassifier(n_estimators=100,n_jobs=-1)
		clf_rf = rf.fit(X_train,y_train)
		score_rf=clf_rf.score(X_Test_Change,y_Test_Change)
		t_end=time()
		t_rndm = t_end - t_start
		print "\n"

		# Calculation for Naive Bayes
		t_start=time()
		print "NaiveBayes"
		nb = BernoulliNB()
		clf_nb=nb.fit(X_train,y_train)
		score_nb=clf_nb.score(X_Test_Change,y_Test_Change)
		t_end=time()
		t_nb = t_end - t_start
		print "\n"

		# Calculation for Logistic Regression
		t_start=time()
		print "Logistic Regression"
		clf = LogisticRegression()
		clf_lr = clf.fit(X_train,y_train)
		score_lr = clf_lr.score(X_Test_Change,y_Test_Change)
		t_end=time()
		t_lr = t_end - t_start
		print "\n"



		# Append all data score and timing to list.
		dataset_perc.append(i/100.0+0.05)
		graph_dt.append(score_dt)
		graph_knn.append(score_knn)
		graph_rndm.append(score_rf)
		graph_nb.append(score_nb)
		graph_lr.append(score_lr)

		time_dt.append(t_dt)
		time_knn.append(t_knn)
		time_rndm.append(t_rndm)
		time_nb.append(t_nb)
		time_lr.append(t_lr)


	# Generate graph for Accuracy versus training data set size.
	trace0 = go.Scatter(x = dataset_perc,y = graph_dt,name = 'Decision Tree',)
	trace1 = go.Scatter(x = dataset_perc,y = graph_knn,name = 'KNN',)
	trace2 = go.Scatter(x = dataset_perc,y = graph_rndm,name = 'Random Forest',)
	trace3 = go.Scatter(x = dataset_perc,y = graph_nb,name = 'Naive Bayes',)
	trace4 = go.Scatter(x = dataset_perc,y = graph_lr,name = 'Logistic Regression',)
	trace5 = go.Scatter(x = dataset_perc,y = graph_svm,name = 'SVM',)
	data = [trace0, trace1, trace2, trace3, trace4]
	layout = dict(title = 'Accuracy as training dataset for different set size',
	              xaxis = dict(title = '% of original training size data'),
	              yaxis = dict(title = 'Accuracy'),)

	fig = dict(data=data, layout=layout)
	py.plot(fig, filename='change-testing-accuracy',auto_open=False)



	# Generate graph for Time versus training data set size.
	tim_0 = go.Scatter(x = dataset_perc,y = time_dt,name = 'DecisionTree')
	tim_1 = go.Scatter(x = dataset_perc,y = time_knn,name = 'KNN')
	tim_2 = go.Scatter(x = dataset_perc,y = time_rndm,name = 'Random Forest')
	tim_3 = go.Scatter(x = dataset_perc,y = time_nb,name = 'NaiveBayes')
	tim_4 = go.Scatter(x = dataset_perc,y = time_lr,name = 'Logistic Regression')

	data1 = [tim_0,tim_1,tim_2,tim_3,tim_4]
	layout1 = dict(title = 'Time for different set size',
	              xaxis = dict(title = '% of original training size data'),
	              yaxis = dict(title = 'Time'),)

	fig1 = dict(data=data1, layout=layout1)
	py.plot(fig1, filename='change-testing-timing-1',auto_open=False)



def run(filename):
	# print filename
	changeTrain(filename)
	print "Success"



if __name__ == '__main__':
	run(sys.argv[1])