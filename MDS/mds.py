import os, sys, getopt, pdb
from numpy import *
from numpy.linalg import *
from numpy.random import *
import pylab as plt

sys.path.append('../DATA_READ/')
from read_data import * 

def mds(d, dimensions = 2):
    """
    Multidimensional Scaling - Given a matrix of interpoint distances,
    find a set of low dimensional points that have similar interpoint
    distances.
    """

    (n,n) = d.shape
    E = (-0.5 * d**2)

    # Use mat to get column and row means to act as column and row means.
    Er = mat(mean(E,1))
    Es = mat(mean(E,0))

    # From Principles of Multivariate Analysis: A User's Perspective (page 107).
    F = array(E - transpose(Er) - Es + mean(E))

    [U, S, V] = svd(F)

    Y = U * sqrt(S)

    return (Y[:,0:dimensions], S)

def norm(vec):
    return sqrt(sum(vec**2))

def square_points(size):
    nsensors = size ** 2
    return array([(i / size, i % size) for i in range(nsensors)])

def main():

    [X,Y]=read_dataset(sys.argv[1])
    points=X
    labels=Y

    (n, d) = points.shape;
    distance = zeros((n,n))
    for (i, pointi) in enumerate(points):
        for (j, pointj) in enumerate(points):
            distance[i,j] = norm(pointi - pointj)

    Y, eigs = mds(distance)
    plt.figure(1)
    classes = list()
    for i in labels:
        classes += [i]
    labels = asarray(classes)
    Y = asarray(Y)
    labels = asarray(labels)
    plt.scatter(Y[:,0],Y[:,1], 20, labels)
    plt.show()

if __name__ ==  "__main__":
    main()

# [X,Y]=read_dataset(sys.argv[1])
# points=X
# labels=Y
# (n, d) = points.shape;
# points = points - tile(mean(points, 0), (n, 1));
# (l, M) = linalg.eig(dot(points.T, points));
# X = dot(points, M[:,0:2]);
# (n, d) = X.shape;
# plt.figure(2)
# plt.scatter(X[:,0], X[:,1], 20, labels);
# plt.show()
