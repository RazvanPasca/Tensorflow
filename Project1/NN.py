import numpy as np
import matplotlib
import PyQt5

from Project1.graph import Graph, Placeholder, Variable
from Project1.operation import *
from Project1.session import Session

matplotlib.use("qt5agg")
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


data = make_blobs(n_samples=50,n_features=2, centers=2,random_state=75)

#we have a tuple with a matrix and a list of features
features = data[0] #get the matrix with x,y values on columns
labels = data[1] #get the labels
print(data)

x = np.linspace(0,11,10)
y = -x + 5
plt.scatter(features[:,0],features[:,1],c = labels,cmap ="coolwarm")
plt.plot(x,y)
# plt.show()

#(1,2)*feaurematrix(2,1) -5 = 0
#weights * features + bias = 0
#we try this for our points, eg 8,10
# print(np.array([1,1]).dot(np.array([[8],[10]])) - 5)

g = Graph()
g.set_as_default()
x = Placeholder()
w = Variable([1,1])
b = Variable(-5)
z = add(matmul(w,x),b)
a = Sigmoid(z)

sess = Session()
sess.run(operation=a,feed_dict={x:[8,10]})