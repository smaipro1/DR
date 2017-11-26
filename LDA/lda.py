import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('../DATA_READ/')
from read_data import *
import math
[X,Y]=read_dataset(sys.argv[1])

s = (784,784)

np.set_printoptions(precision=10)

mean_vectors = []
for cl in range(0,10):
    mean_vectors.append(np.mean(X[Y==cl], axis=0))
    #print('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl-1]))



S_W = np.zeros(s)
for cl,mv in zip(range(0,10), mean_vectors):
    class_sc_mat = np.zeros(s)                  # scatter matrix for every class
    for row in X[Y == cl]:
        row, mv = row.reshape(sample_length,1), mv.reshape(sample_length,1) # make column vectors
        class_sc_mat += np.dot((row-mv),(row-mv).T)
    S_W += class_sc_mat                             # sum class scatter matrices
print('within-class Scatter Matrix:\n', S_W)
# for i in S_W:
	# print i
print np.linalg.det(S_W)

overall_mean = np.mean(X, axis=0)

S_B = np.zeros(s)
for i,mean_vec in enumerate(mean_vectors):  
    n = X[Y==i,:].shape[0]
    mean_vec = mean_vec.reshape(sample_length,1) # make column vector
    overall_mean = overall_mean.reshape(sample_length,1) # make column vector
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

print('between-class Scatter Matrix:\n', S_B)
S = np.linalg.inv(S_W)*S_B