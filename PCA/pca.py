import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('../DATA_READ/')
from read_data import *

[X,Y]=read_dataset()

print X.shape
u,s,v = np.linalg.svd(X.T)
red = u[0:sample_length/4,:]
X_red = np.dot(X,red.T)
# for i in X_red:
    # print i
print X_red.shape

matrix = X[0].reshape((28,28))
plt.imshow(matrix,cmap='gray') 
plt.savefig('HELLO1')


matrix_red = X_red[0].reshape((14, 14))
plt.imshow(matrix_red,cmap='gray') 
plt.savefig('HELLO2')