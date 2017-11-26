import os, sys, getopt, pdb
from numpy import *
from numpy.linalg import *
from numpy.random import *
import pylab as plt

def main():

    points = loadtxt("mnist2500_X.txt");
    labels = loadtxt("mnist2500_labels.txt");

    (n, d) = points.shape;
    points = points - tile(mean(points, 0), (n, 1));
    (l, M) = linalg.eig(dot(points.T, points));
    X = dot(points, M[:,0:2]);
    (n, d) = X.shape;

    plt.scatter(X[:,0], X[:,1], 20, labels);

    plt.show()

if __name__ == "__main__":
    main()
import pylab as plt