import pandas as pd

def read_dataset():
	sample_length=784
	red_sample_length=sample_length/4
	df = pd.read_csv(
	    filepath_or_buffer='~/DATASET/dataset.csv', 
	    header=None, 
	    sep=',')

	X = df.ix[:,1:sample_length].values
	Y = df.ix[:,0]
	return [X,Y]
	
