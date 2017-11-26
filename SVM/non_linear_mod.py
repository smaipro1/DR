import sys
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import time
############################################one versus one kernel########################

def handler(decision_function_shape_in,gamma_in,dataset_in):
	#non linear kernel for svm classifier 
	
	sys.path.append('../DATA_READ/')
	from read_data import *	
	
	[X,Y]=read_dataset(dataset_in)
	X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)
	

	print X_train.shape
	print X_test.shape
	#rbf kernel 
	
	time_array=[]

	t0 = time.time()

	rbf_svc = svm.SVC(decision_function_shape=decision_function_shape_in,gamma=gamma_in,kernel='rbf')
	rbf_svc.fit(X_train,Y_train)
	Y_predict_rbf=rbf_svc.predict(X_test)  
	accr_rbf_svc=confusion_matrix(Y_predict_rbf, Y_test)
	accr_rbf_svc_score=accuracy_score(Y_predict_rbf, Y_test)
	print accr_rbf_svc_score

	t1 = time.time()	
	temp_time=t1-t0
	time_array.append(temp_time)
	t0 = time.time()


	#polynomial degree 2
	polynomial_svc_2=svm.SVC(decision_function_shape=decision_function_shape_in,gamma=gamma_in,degree=2,kernel='poly')
	polynomial_svc_2.fit(X_train,Y_train)
	Y_predict_polynomial_svc_2=polynomial_svc_2.predict(X_test)  
	accr_polynomial_svc_2=confusion_matrix(Y_predict_polynomial_svc_2, Y_test)
	accr_polynomial_svc_2_score=accuracy_score(Y_predict_polynomial_svc_2, Y_test)
	print accr_polynomial_svc_2_score

	t1 = time.time()	
	temp_time=t1-t0
	time_array.append(temp_time)
	t0 = time.time()




	#polynomial degree 3
	polynomial_svc_3=svm.SVC(decision_function_shape=decision_function_shape_in,gamma=gamma_in,degree=3,kernel='poly')
	polynomial_svc_3.fit(X_train,Y_train)
	Y_predict_polynomial_svc_3=polynomial_svc_3.predict(X_test)  
	accr_polynomial_svc_3=confusion_matrix(Y_predict_polynomial_svc_3, Y_test)
	accr_polynomial_svc_3_score=accuracy_score(Y_predict_polynomial_svc_3, Y_test)
	print accr_polynomial_svc_3_score

	t1 = time.time()	
	temp_time=t1-t0
	time_array.append(temp_time)
	t0 = time.time()



	#sigmoid kernel
	sigmoid_svc = svm.SVC(decision_function_shape=decision_function_shape_in,gamma=gamma_in,kernel='sigmoid')
	sigmoid_svc.fit(X_train,Y_train)
	Y_predict_sigmoid_svc=sigmoid_svc.predict(X_test)  
	accr_sigmoid_svc=confusion_matrix(Y_predict_sigmoid_svc, Y_test)
	accr_sigmoid_svc_score=accuracy_score(Y_predict_sigmoid_svc, Y_test)
	print accr_sigmoid_svc_score

	t1 = time.time()	
	temp_time=t1-t0
	time_array.append(temp_time)
	
	print 'time array is ' , time_array
	
	#return [accr_rbf_svc_score,accr_polynomial_svc_3_score,accr_polynomial_svc_4_score,accr_polynomial_svc_5_score,accr_polynomial_svc_6_score,accr_polynomial_svc_7_score,accr_polynomial_svc_8_score,accr_polynomial_svc_9_score,accr_polynomial_svc_10_score,accr_sigmoid_svc_score]
	return [time_array,[accr_rbf_svc_score,accr_polynomial_svc_2_score,accr_polynomial_svc_3_score,accr_sigmoid_svc_score]]

def non_linear_train(dataset_in):
	
	#gamma values varied
	gamma_values=[1,10,50,100,250]
	rows = ['rbf','polynomial2','polynomial3','polynomial4','polynomial5','polynomial6','polynomial7','polynomial8','polynomial9','polynomial10','Sigmoid']
	columns = ['Gamma Values = %d ' % x for x in gamma_values]
	result_matrix=[]
	main_time_array=[]
	#############################################################################################################
	#one versus one svm
	
	for j in gamma_values:
		[time_array,temp]=handler('ovo',j,dataset_in)
		result_matrix.append(temp)
		main_time_array.append(time_array)	
	np.savetxt('non linear one versus one.csv',result_matrix,delimiter=',',fmt='%.6f')
	np.savetxt('non linear one versus one_time.csv',main_time_array,delimiter=',',fmt='%.6f')

	############################################################################################################
	#one versus other svm
	for j in gamma_values:
		[time_array,temp]=handler('ovo',j,dataset_in)
		result_matrix.append(temp)
		main_time_array.append(time_array)	
	np.savetxt('non linear one versus rest.csv',result_matrix,delimiter=',',fmt='%.6f')
	np.savetxt('non linear one versus rest_time.csv',main_time_array,delimiter=',',fmt='%.6f')	
	#############################################################################################################