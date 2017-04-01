# Title		 	: change_TrainData.py
# Description 	: This file generating acurrancy and timing graph by changing in Training Dataset.
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
from sklearn import svm
from sklearn.grid_search import GridSearchCV
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
	This compare diffnet alogrithm for data in input file. 
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
	graph_svm = []

	time_knn =[]
	time_dt = []
	time_rndm = []
	time_nb = []
	time_lr = []
	time_svm = []

	# Comparison of all alorithm and measuure accurancy and time.
	# Increasingly add size of training set 5% of orginal, keep testing size unchanged
	for i in range(0,100,5):
		X_rest, X_Train_Change, y_rest, y_Train_Change = cross_validation.train_test_split(X_train, y_train, test_size=0.049+i/100.0, random_state=0)
		print "------ Loop for training size is %s perc of original data --------- " %i 

		# Calculation for Decision Tree
		t_start=time()
		print "DecisionTree"
		dt = DecisionTreeClassifier(min_samples_split=20,random_state=99)
		clf_dt = dt.fit(X_Train_Change,y_Train_Change)
		score_dt = clf_dt.score(X_test,y_test)
		t_end=time()
		t_dt = t_end - t_start
		print "\n"

		# Calculation for KNN
		t_start=time()
		print "KNN"
		knn = KNeighborsClassifier()
		clf_knn=knn.fit(X_Train_Change, y_Train_Change)
		score_knn=clf_knn.score(X_test,y_test)
		t_end=time()
		t_knn=t_end - t_start
		print "\n"

		# Calculation for Random Forest
		t_start=time()
		print "RandomForest"
		rf = RandomForestClassifier(n_estimators=100,n_jobs=-1)
		clf_rf = rf.fit(X_Train_Change,y_Train_Change)
		score_rf=clf_rf.score(X_test,y_test)
		t_end=time()
		t_rndm = t_end - t_start
		print "\n"

		# Calculation for Naive Bayes
		t_start=time()
		print "NaiveBayes"
		nb = BernoulliNB()
		clf_nb=nb.fit(X_Train_Change,y_Train_Change)
		score_nb=clf_nb.score(X_test,y_test)
		t_end=time()
		t_nb = t_end - t_start
		print "\n"

		# Calculation for Logistic Regression
		t_start=time()
		print "Logistic Regression"
		clf = LogisticRegression()
		clf_lr = clf.fit(X_Train_Change, y_Train_Change)
		score_lr = clf_lr.score(X_test,y_test)
		t_end=time()
		t_lr = t_end - t_start
		print "\n"

		# # Calculation for SVM
		# t_start=time()
		# print "SVM"
		# svc = svm.SVC(kernel='linear')
		# clf = GridSearchCV(estimator=svc, param_grid=dict(C=[5]), cv = 10)
		# clf_svm = clf.fit(X_Train_Change, y_Train_Change)
		# score_svm = clf_svm.score(X_test,y_test)
		# t_end=time()
		# t_svm = t_end - t_start
		# print "\n"


		# Append all data score and timing to list.
		dataset_perc.append(i/100.0+0.05)
		graph_dt.append(score_dt)
		graph_knn.append(score_knn)
		graph_rndm.append(score_rf)
		graph_nb.append(score_nb)
		graph_lr.append(score_lr)
		# graph_svm.append(score_svm)

		time_dt.append(t_dt)
		time_knn.append(t_knn)
		time_rndm.append(t_rndm)
		time_nb.append(t_nb)
		time_lr.append(t_lr)
		# time_svm.append(t_svm)

	# Selection of Important feature based on Random Forest 
	imp_f(rf,features)


	# Generate graph for Accuracy versus training data set size.
	trace0 = go.Scatter(x = dataset_perc,y = graph_dt,name = 'Decision Tree',)
	trace1 = go.Scatter(x = dataset_perc,y = graph_knn,name = 'KNN',)
	trace2 = go.Scatter(x = dataset_perc,y = graph_rndm,name = 'Random Forest',)
	trace3 = go.Scatter(x = dataset_perc,y = graph_nb,name = 'Naive Bayes',)
	trace4 = go.Scatter(x = dataset_perc,y = graph_lr,name = 'Logistic Regression',)
	trace5 = go.Scatter(x = dataset_perc,y = graph_svm,name = 'SVM',)

	data = [trace0, trace1, trace2, trace3, trace4, trace5]
	layout = dict(title = 'Accuracy as training dataset for different set size',
	              xaxis = dict(title = 'Percent of original training size data'),
	              yaxis = dict(title = 'Accuracy based (.score)'),)

	fig = dict(data=data, layout=layout)
	py.iplot(fig, filename='change-training-accuracy',auto_open=False)



	# Generate graph for Time versus training data set size.
	tim_0 = go.Scatter(x = dataset_perc,y = time_dt,name = 'Decision Tree')
	tim_1 = go.Scatter(x = dataset_perc,y = time_knn,name = 'KNN')
	tim_2 = go.Scatter(x = dataset_perc,y = time_rndm,name = 'Random Forest')
	tim_3 = go.Scatter(x = dataset_perc,y = time_nb,name = 'Naive Bayes')
	tim_4 = go.Scatter(x = dataset_perc,y = time_lr,name = 'Logistic Regression')

	data1 = [tim_0,tim_1,tim_2,tim_3,tim_4]
	layout = dict(title = 'Time taken for different training set size',
	              xaxis = dict(title = 'Percent of training size data'),
	              yaxis = dict(title = 'Time (s)'),)

	fig = dict(data=data1, layout=layout)
	py.plot(fig, filename='change-training-timing',auto_open=False)



# Important feature selection using randomforest feature importances method
def imp_f(rf,features):
	print "Features sorted by their score:"

	# http://blog.datadive.net/selecting-good-features-part-iii-random-forests/
	imp_features = sorted(zip(map(lambda x: round(x,5), rf.feature_importances_), features), reverse=True)
	print imp_features
	impfeatures_dic = {}
	for i,j in imp_features:
		impfeatures_dic[j] = i

	# Print import features sorted with dicitonary values.
	# print sorted(impfeatures_dic.values(),reverse=True)
	# print sorted(impfeatures_dic, key=impfeatures_dic.get,reverse=True)

	data = [go.Bar(x=sorted(impfeatures_dic, key=impfeatures_dic.get,reverse=True),y=sorted(impfeatures_dic.values(),reverse=True))]
	layout = dict(title = 'Random forest feature importance',
	              xaxis = dict(title = 'Attributes'),
	              yaxis = dict(title = ''),)
	fig = go.Figure(data=data, layout=layout)
	plot_url = py.plot(fig, filename='Attributes Importancee', auto_open=False)





def run(filename):
	# print filename
	changeTrain(filename)
	print 'Success'



if __name__ == '__main__':
	run(sys.argv[1])