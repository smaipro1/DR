import sys

sys.path.append('../DATA_READ/')
from read_data import *

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

[X,Y]=read_dataset()
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)


print X_train.shape
print X_test.shape


rbf_svc = svm.SVC(decision_function_shape='ovo',kernel='rbf')
rbf_svc.fit(X_train,Y_train)
Y_predict_rbf=rbf_svc.predict(X_test)  
accr_rbf_svc=confusion_matrix(Y_predict_rbf, Y_test)
accr_rbf_svc_score=accuracy_score(Y_predict_rbf, Y_test)
print accr_rbf_svc_score


polynomial_svc_2=svm.SVC(decision_function_shape='ovo',degree=2,kernel='poly')
polynomial_svc_2.fit(X_train,Y_train)
Y_predict_polynomial_svc_2=polynomial_svc_2.predict(X_test)  
accr_polynomial_svc_2=confusion_matrix(Y_predict_polynomial_svc_2, Y_test)
accr_polynomial_svc_2_score=accuracy_score(Y_predict_polynomial_svc_2, Y_test)
print accr_polynomial_svc_2_score

polynomial_svc_3=svm.SVC(decision_function_shape='ovo',degree=3,kernel='poly')
polynomial_svc_3.fit(X_train,Y_train)
Y_predict_polynomial_svc_3=polynomial_svc_3.predict(X_test)  
accr_polynomial_svc_3=confusion_matrix(Y_predict_polynomial_svc_3, Y_test)
accr_polynomial_svc_3_score=accuracy_score(Y_predict_polynomial_svc_3, Y_test)
print accr_polynomial_svc_3_score

polynomial_svc_4=svm.SVC(decision_function_shape='ovo',degree=4,kernel='poly')
polynomial_svc_4.fit(X_train,Y_train)
Y_predict_polynomial_svc_4=polynomial_svc_4.predict(X_test)  
accr_polynomial_svc_4=confusion_matrix(Y_predict_polynomial_svc_4, Y_test)
accr_polynomial_svc_4_score=accuracy_score(Y_predict_polynomial_svc_4, Y_test)
print accr_polynomial_svc_4_score

polynomial_svc_5=svm.SVC(decision_function_shape='ovo',degree=5,kernel='poly')
polynomial_svc_5.fit(X_train,Y_train)
Y_predict_polynomial_svc_5=polynomial_svc_5.predict(X_test)  
accr_polynomial_svc_5=confusion_matrix(Y_predict_polynomial_svc_5, Y_test)
accr_polynomial_svc_5_score=accuracy_score(Y_predict_polynomial_svc_5, Y_test)
print accr_polynomial_svc_5_score

polynomial_svc_6=svm.SVC(decision_function_shape='ovo',degree=6,kernel='poly')
polynomial_svc_6.fit(X_train,Y_train)
Y_predict_polynomial_svc_6=polynomial_svc_6.predict(X_test)  
accr_polynomial_svc_6=confusion_matrix(Y_predict_polynomial_svc_6, Y_test)
accr_polynomial_svc_6_score=accuracy_score(Y_predict_polynomial_svc_6, Y_test)
print accr_polynomial_svc_6_score

polynomial_svc_7=svm.SVC(decision_function_shape='ovo',degree=7,kernel='poly')
polynomial_svc_7.fit(X_train,Y_train)
Y_predict_polynomial_svc_7=polynomial_svc_7.predict(X_test)  
accr_polynomial_svc_7=confusion_matrix(Y_predict_polynomial_svc_7, Y_test)
accr_polynomial_svc_7_score=accuracy_score(Y_predict_polynomial_svc_7, Y_test)
print accr_polynomial_svc_7_score

polynomial_svc_8=svm.SVC(decision_function_shape='ovo',degree=8,kernel='poly')
polynomial_svc_8.fit(X_train,Y_train)
Y_predict_polynomial_svc_8=polynomial_svc_8.predict(X_test)  
accr_polynomial_svc_8=confusion_matrix(Y_predict_polynomial_svc_8, Y_test)
accr_polynomial_svc_8_score=accuracy_score(Y_predict_polynomial_svc_8, Y_test)
print accr_polynomial_svc_8_score

polynomial_svc_9=svm.SVC(decision_function_shape='ovo',degree=9,kernel='poly')
polynomial_svc_9.fit(X_train,Y_train)
Y_predict_polynomial_svc_9=polynomial_svc_9.predict(X_test)  
accr_polynomial_svc_9=confusion_matrix(Y_predict_polynomial_svc_9, Y_test)
accr_polynomial_svc_9_score=accuracy_score(Y_predict_polynomial_svc_9, Y_test)
print accr_polynomial_svc_9_score

polynomial_svc_10=svm.SVC(decision_function_shape='ovo',degree=10,kernel='poly')
polynomial_svc_10.fit(X_train,Y_train)
Y_predict_polynomial_svc_10=polynomial_svc_10.predict(X_test)  
accr_polynomial_svc_10=confusion_matrix(Y_predict_polynomial_svc_10, Y_test)
accr_polynomial_svc_10_score=accuracy_score(Y_predict_polynomial_svc_10, Y_test)
print accr_polynomial_svc_10_score

sigmoid_svc = svm.SVC(decision_function_shape='ovo',kernel='sigmoid')
sigmoid_svc.fit(X_train,Y_train)
Y_predict_sigmoid_svc=sigmoid_svc.predict(X_test)  
accr_sigmoid_svc=confusion_matrix(Y_predict_sigmoid_svc, Y_test)
accr_sigmoid_svc_score=accuracy_score(Y_predict_sigmoid_svc, Y_test)
print accr_sigmoid_svc_score

lin_clf = svm.LinearSVC()
lin_clf.fit(X_train,Y_train)
Y_lin_clf=lin_clf.predict(X_test)  
accr_lin_clf=confusion_matrix(Y_lin_clf, Y_test)
accr_lin_clf_score=accuracy_score(Y_lin_clf, Y_test)
print accr_lin_clf_score
