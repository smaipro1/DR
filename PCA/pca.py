#this is pca code
sample_length=784
red_sample_length=sample_length/4
import pandas as pd
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
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)


import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls
import numpy as np

cor_mat1 = np.corrcoef(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cor_mat1)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()


tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)


trace1 = Bar(
        x=['PC %s' %i for i in range(1,sample_length)],
        y=var_exp,
        showlegend=False)

trace2 = Scatter(
        x=['PC %s' %i for i in range(1,sample_length)], 
        y=cum_var_exp,
        name='cumulative explained variance')

data = Data([trace1, trace2])

layout=Layout(
        yaxis=YAxis(title='Explained variance in percent'),
        title='Explained variance by different principal components')

fig = Figure(data=data, layout=layout)
py.iplot(fig)
exit()
temp_mat=[]
for i in range(1,red_sample_length+1):
	temp_mata.append((eig_pairs[i-1][1].reshape(sample_length,1)))
	
matrix_w = np.hstack(temp_mat)

print('Matrix W:\n', matrix_w)

Y = X_std.dot(matrix_w)


traces = []

'''for name in ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'):

    trace = Scatter(
        x=Y[y==name,0],
        y=Y[y==name,1],
        mode='markers',
        name=name,
        marker=Marker(
            size=12,
            line=Line(
                color='rgba(217, 217, 217, 0.14)',
                width=0.5),
            opacity=0.8))
    traces.append(trace)


data = Data(traces)
layout = Layout(showlegend=True,
                scene=Scene(xaxis=XAxis(title='PC1'),
                yaxis=YAxis(title='PC2'),))

fig = Figure(data=data, layout=layout)
py.iplot(fig)

'''
