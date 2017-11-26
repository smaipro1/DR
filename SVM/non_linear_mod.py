import sys
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import plotly.plotly as py
import plotly.graph_objs as go

############################################one versus one kernel########################

def handler(decision_function_shape_in,gamma_in,dataset_in):
	sys.path.append('../DATA_READ/')
	from read_data import *
	
	[X,Y]=read_dataset(dataset_in)
	X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)
	

	print X_train.shape
	print X_test.shape
	
	rbf_svc = svm.SVC(decision_function_shape=decision_function_shape_in,gamma=gamma_in,kernel='rbf')
	rbf_svc.fit(X_train,Y_train)
	Y_predict_rbf=rbf_svc.predict(X_test)  
	accr_rbf_svc=confusion_matrix(Y_predict_rbf, Y_test)
	accr_rbf_svc_score=accuracy_score(Y_predict_rbf, Y_test)
	print accr_rbf_svc_score


	polynomial_svc_2=svm.SVC(decision_function_shape=decision_function_shape_in,gamma=gamma_in,degree=2,kernel='poly')
	polynomial_svc_2.fit(X_train,Y_train)
	Y_predict_polynomial_svc_2=polynomial_svc_2.predict(X_test)  
	accr_polynomial_svc_2=confusion_matrix(Y_predict_polynomial_svc_2, Y_test)
	accr_polynomial_svc_2_score=accuracy_score(Y_predict_polynomial_svc_2, Y_test)
	print accr_polynomial_svc_2_score

	polynomial_svc_3=svm.SVC(decision_function_shape=decision_function_shape_in,gamma=gamma_in,degree=3,kernel='poly')
	polynomial_svc_3.fit(X_train,Y_train)
	Y_predict_polynomial_svc_3=polynomial_svc_3.predict(X_test)  
	accr_polynomial_svc_3=confusion_matrix(Y_predict_polynomial_svc_3, Y_test)
	accr_polynomial_svc_3_score=accuracy_score(Y_predict_polynomial_svc_3, Y_test)
	print accr_polynomial_svc_3_score

	polynomial_svc_4=svm.SVC(decision_function_shape=decision_function_shape_in,gamma=gamma_in,degree=4,kernel='poly')
	polynomial_svc_4.fit(X_train,Y_train)
	Y_predict_polynomial_svc_4=polynomial_svc_4.predict(X_test)  
	accr_polynomial_svc_4=confusion_matrix(Y_predict_polynomial_svc_4, Y_test)
	accr_polynomial_svc_4_score=accuracy_score(Y_predict_polynomial_svc_4, Y_test)
	print accr_polynomial_svc_4_score

	polynomial_svc_5=svm.SVC(decision_function_shape=decision_function_shape_in,gamma=gamma_in,degree=5,kernel='poly')
	polynomial_svc_5.fit(X_train,Y_train)
	Y_predict_polynomial_svc_5=polynomial_svc_5.predict(X_test)  
	accr_polynomial_svc_5=confusion_matrix(Y_predict_polynomial_svc_5, Y_test)
	accr_polynomial_svc_5_score=accuracy_score(Y_predict_polynomial_svc_5, Y_test)
	print accr_polynomial_svc_5_score

	polynomial_svc_6=svm.SVC(decision_function_shape=decision_function_shape_in,gamma=gamma_in,degree=6,kernel='poly')
	polynomial_svc_6.fit(X_train,Y_train)
	Y_predict_polynomial_svc_6=polynomial_svc_6.predict(X_test)  
	accr_polynomial_svc_6=confusion_matrix(Y_predict_polynomial_svc_6, Y_test)
	accr_polynomial_svc_6_score=accuracy_score(Y_predict_polynomial_svc_6, Y_test)
	print accr_polynomial_svc_6_score

	polynomial_svc_7=svm.SVC(decision_function_shape=decision_function_shape_in,gamma=gamma_in,degree=7,kernel='poly')
	polynomial_svc_7.fit(X_train,Y_train)
	Y_predict_polynomial_svc_7=polynomial_svc_7.predict(X_test)  
	accr_polynomial_svc_7=confusion_matrix(Y_predict_polynomial_svc_7, Y_test)
	accr_polynomial_svc_7_score=accuracy_score(Y_predict_polynomial_svc_7, Y_test)
	print accr_polynomial_svc_7_score

	polynomial_svc_8=svm.SVC(decision_function_shape=decision_function_shape_in,gamma=gamma_in,degree=8,kernel='poly')
	polynomial_svc_8.fit(X_train,Y_train)
	Y_predict_polynomial_svc_8=polynomial_svc_8.predict(X_test)  
	accr_polynomial_svc_8=confusion_matrix(Y_predict_polynomial_svc_8, Y_test)
	accr_polynomial_svc_8_score=accuracy_score(Y_predict_polynomial_svc_8, Y_test)
	print accr_polynomial_svc_8_score

	polynomial_svc_9=svm.SVC(decision_function_shape=decision_function_shape_in,gamma=gamma_in,degree=9,kernel='poly')
	polynomial_svc_9.fit(X_train,Y_train)
	Y_predict_polynomial_svc_9=polynomial_svc_9.predict(X_test)  
	accr_polynomial_svc_9=confusion_matrix(Y_predict_polynomial_svc_9, Y_test)
	accr_polynomial_svc_9_score=accuracy_score(Y_predict_polynomial_svc_9, Y_test)
	print accr_polynomial_svc_9_score

	polynomial_svc_10=svm.SVC(decision_function_shape=decision_function_shape_in,gamma=gamma_in,degree=10,kernel='poly')
	polynomial_svc_10.fit(X_train,Y_train)
	Y_predict_polynomial_svc_10=polynomial_svc_10.predict(X_test)  
	accr_polynomial_svc_10=confusion_matrix(Y_predict_polynomial_svc_10, Y_test)
	accr_polynomial_svc_10_score=accuracy_score(Y_predict_polynomial_svc_10, Y_test)
	print accr_polynomial_svc_10_score

	sigmoid_svc = svm.SVC(decision_function_shape=decision_function_shape_in,gamma=gamma_in,kernel='sigmoid')
	sigmoid_svc.fit(X_train,Y_train)
	Y_predict_sigmoid_svc=sigmoid_svc.predict(X_test)  
	accr_sigmoid_svc=confusion_matrix(Y_predict_sigmoid_svc, Y_test)
	accr_sigmoid_svc_score=accuracy_score(Y_predict_sigmoid_svc, Y_test)
	print accr_sigmoid_svc_score
	
	return [accr_rbf_svc_score,accr_polynomial_svc_3_score,accr_polynomial_svc_4_score,accr_polynomial_svc_5_score,accr_polynomial_svc_6_score,accr_polynomial_svc_7_score,accr_polynomial_svc_8_score,accr_polynomial_svc_9_score,accr_polynomial_svc_10_score,accr_sigmoid_svc_score]

def non_linear_train(dataset_in):
	gamma_values=[1,10,50,100,250]
	columns = ['rbf','polynomial2','polynomial3','polynomial4','polynomial5','polynomial6','polynomial7','polynomial8','polynomial9','polynomial10','Sigmoid']
	rows = ['Gamma Values = %d ' % x for x in gamma_values]
	
	
	values1=[]

	#############################################################################################################
	#one versus one kernels
	for j in gamma_values:
		temp=handler('ovo',j,dataset_in)
		values1.append([temp])
	#############################################################################################################
	headerColor = 'grey'
	rowEvenColor = 'lightgrey'
	rowOddColor = 'white'

	trace0 = go.Table(
	  type = 'table',
	  header = dict(
		values = columns,
		line = dict(color = '#506784'),
		fill = dict(color = headerColor),
		align = ['left','center'],
		font = dict(color = 'white', size = 12)
	  ),
	  cells = dict(
		values = [[rows],values1],
		line = dict(color = '#506784'),
		fill = dict(color = [[rowOddColor,rowEvenColor,rowOddColor,
								   rowEvenColor,rowOddColor]]),
		align = ['left', 'center'],
		font = dict(color = '#506784', size = 11)
		))

	data = [trace0]
	py.iplot(data, filename = "non linear ovo")

	#one versus rest kernels
	values=[]

	#############################################################################################################
	#one versus other kernels
	for j in gamma_values:
		temp=handler('ovr',j,dataset_in)
		values.append([temp])
	#############################################################################################################
	headerColor = 'grey'
	rowEvenColor = 'lightgrey'
	rowOddColor = 'white'

	trace0 = go.Table(
	  type = 'table',
	  header = dict(
		values = columns,
		line = dict(color = '#506784'),
		fill = dict(color = headerColor),
		align = ['left','center'],
		font = dict(color = 'white', size = 12)
	  ),
	  cells = dict(
		values = [[rows],values],
		line = dict(color = '#506784'),
		fill = dict(color = [[rowOddColor,rowEvenColor,rowOddColor,
								   rowEvenColor,rowOddColor]]),
		align = ['left', 'center'],
		font = dict(color = '#506784', size = 11)
		))

	data = [trace0]
	py.iplot(data, filename = "non linear ovr")
