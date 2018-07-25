import numpy as np
import matplotlib.pyplot as plt


from Project1.graph import Graph, Variable, Placeholder
from Project1.operation import add, matmul
from Project1.session import Session


def test():
   g = Graph()
   g.set_as_default()
   A = Variable([[10,20],[30,40]])
   b = Variable([1,2])
   x = Placeholder()
   y = matmul(A,x)
   z = add(y,b)

   sess = Session()
   print(sess.run(operation= z, feed_dict={x:10}))
   print(np.array([[10,20],[30,40]]).dot(np.eye(2,2)))

test()