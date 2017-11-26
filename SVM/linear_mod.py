import sys
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#########################################################################################
#########Linear Support Vector Classification.#######################
def linear_train(dataset_in):
	values1=[]
	
	sys.path.append('../DATA_READ/')
	from read_data import *
	
	[X,Y]=read_dataset(dataset_in)
	X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)
	
	print X_train.shape
	print X_test.shape
	
	penalty_temp='l1'
	loss_temp='hinge'

	lin_clf = svm.LinearSVC(penalty=penalty_temp,loss=loss_temp)
	lin_clf.fit(X_train,Y_train)
	Y_lin_clf=lin_clf.predict(X_test)  
	accr_lin_clf=confusion_matrix(Y_lin_clf, Y_test)
	accr_lin_clf_score=accuracy_score(Y_lin_clf, Y_test)
	values1=values1.append(accr_lin_clf_score)
	print accr_lin_clf_score




	penalty_temp='l1'
	loss_temp='squared_hinge'


	lin_clf = svm.LinearSVC(penalty=penalty_temp,loss=loss_temp)
	lin_clf.fit(X_train,Y_train)
	Y_lin_clf=lin_clf.predict(X_test)  
	accr_lin_clf=confusion_matrix(Y_lin_clf, Y_test)
	accr_lin_clf_score=accuracy_score(Y_lin_clf, Y_test)
	values1=values1.append(accr_lin_clf_score)
	print accr_lin_clf_score

	penalty_temp='l2'
	loss_temp='hinge'


	lin_clf = svm.LinearSVC(penalty=penalty_temp,loss=loss_temp)
	lin_clf.fit(X_train,Y_train)
	Y_lin_clf=lin_clf.predict(X_test)  
	accr_lin_clf=confusion_matrix(Y_lin_clf, Y_test)
	accr_lin_clf_score=accuracy_score(Y_lin_clf, Y_test)
	values1=values1.append(accr_lin_clf_score)
	print accr_lin_clf_score

	penalty_temp='l2'
	loss_temp='squared_hinge'

	lin_clf = svm.LinearSVC(penalty=penalty_temp,loss=loss_temp)
	lin_clf.fit(X_train,Y_train)
	Y_lin_clf=lin_clf.predict(X_test)  
	accr_lin_clf=confusion_matrix(Y_lin_clf, Y_test)
	accr_lin_clf_score=accuracy_score(Y_lin_clf, Y_test)
	values1=values1.append(accr_lin_clf_score)
	print accr_lin_clf_score

	rows = ['l1 norm','l2 norm']
	columns = ['hinge loss','squared hinge loss']
	
		
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
	py.iplot(data, filename = "linear")
