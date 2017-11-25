import pandas as pd
sample_length=784
def read_dataset():

	red_sample_length=sample_length/4
	df = pd.read_csv(
	#filepath_or_buffer='~/DATASET/dataset.csv',
	#filepath_or_buffer='~/DATASET/mnist_test.csv',
	filepath_or_buffer='~/DATASET/pca_reduced_data.csv', 
	    header=None, 
	    sep=',')

	X = df.ix[:,1:sample_length].values
	Y = df.ix[:,0]
	return [X,Y]
	
