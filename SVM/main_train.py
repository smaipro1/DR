import sys
from linear_mod import *
from non_linear_mod import *
import numpy as np
print '#argument 1 is for name of the dataset'
##########takes the name of the dataset to compute to train svm on linear and non linear kernels
linear_train(sys.argv[1])
non_linear_train(sys.argv[1])
