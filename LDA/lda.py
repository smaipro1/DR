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