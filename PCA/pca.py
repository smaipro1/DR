import numpy as np
import pandas as pd

#this is pca code
sample_length=784
red_sample_length=sample_length/4

#filepath_or_buffer='~/DATASET/mnist_test.csv',
df = pd.read_csv(
    filepath_or_buffer='~/DATASET/testdata.csv', 
    header=None, 
    sep=',')

df.columns=range(0,sample_length+1)
df.dropna(how="all", inplace=True) # drops the empty line at file-end
df.tail()

# split data table into data X and class labels y

X = df.ix[:,1:sample_length].values
y = df.ix[:,0].values

X_std=X
u,s,v = np.linalg.svd(X_std.T)
red = u[0:sample_length/4,:]
X_red = np.dot(X_std,red.T)
for i in X_red:
    print i
