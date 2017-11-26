import sys
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import time

#########################################################################################
#########Linear Support Vector Classification.#######################
def linear_train(dataset_in):
	
	np.set_printoptions(precision=4,suppress=True)
	#linear kernel for svm classifier 
	
	sys.path.append('../DATA_READ/')
	from read_data import *	
	
	####get the dataset##########
	[X,Y]=read_dataset(dataset_in)
	#splits the data into 80 % train and 20 % test
	X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)
	
	print X_train.shape
	print X_test.shape
	
	####################

	time_array=[]

	t0 = time.time()
	############################################uses svm loss hinge#########################################
	loss_temp='hinge'
	result_array=[]

	lin_clf = svm.LinearSVC(loss=loss_temp)
	lin_clf.fit(X_train,Y_train)
	Y_lin_clf=lin_clf.predict(X_test)  
	accr_lin_clf=confusion_matrix(Y_lin_clf, Y_test)
	accr_lin_clf_score=accuracy_score(Y_lin_clf, Y_test)
	result_array.append(accr_lin_clf_score)	
	print accr_lin_clf_score
	
	t1 = time.time()	
	
	temp_time=t1-t0
	time_array.append(temp_time)

	############################################uses svm squared loss hinge#########################################
	

	t0 = time.time()
	loss_temp='squared_hinge'
	lin_clf = svm.LinearSVC(loss=loss_temp)
	lin_clf.fit(X_train,Y_train)
	Y_lin_clf=lin_clf.predict(X_test)  
	accr_lin_clf=confusion_matrix(Y_lin_clf, Y_test)
	accr_lin_clf_score=accuracy_score(Y_lin_clf, Y_test)
	result_array.append(accr_lin_clf_score)
	print accr_lin_clf_score
	t1 = time.time()	
	
	temp_time=t1-t0
	time_array.append(temp_time)	
	np.savetxt('linear.csv',result_array,delimiter=',',fmt='%.6f')	

	np.savetxt('linear_time.csv',time_array,delimiter=',',fmt='%.6f')

