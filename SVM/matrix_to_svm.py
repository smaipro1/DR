import numpy as np
def matrix_converter(matrix,label,name):
	a,b=matrix.shape
	temp_array=[]
	temp_matrix=[]
	print a,b
	for i in range(a):
		temp_array=[]
		temp_array.append(label[i])
		for j in range(b):
			temp_array.append(matrix[i][j])
		temp_matrix.append(temp_array)

	
	temp_matrix=np.array(temp_matrix)
	#print temp_matrix.shape
	np.savetxt(name, temp_matrix, delimiter=",")
	
