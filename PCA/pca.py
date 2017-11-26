import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('../DATA_READ/')
from read_data import *

datain=str(sys.argv[1])

[X,Y]=read_dataset(datain)
u,s,v = np.linalg.svd(X.T)
red = u[0:sample_length/4,:]
X_red = np.dot(X,red.T)

sys.path.append('../SVM/')
from matrix_to_svm import *

matrix_converter(X_red,Y)



'''matrix = X[0].reshape((28,28))
plt.imshow(matrix,cmap='gray') 
plt.savefig('HELLO1')


matrix_red = X_red[0].reshape((14, 14))
plt.imshow(matrix_red,cmap='gray') 
plt.savefig('HELLO2')'''
