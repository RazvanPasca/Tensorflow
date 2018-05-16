import tensorflow as tf
#graphs are sets of connected nodes
#we have edges and nodes
#A node is an operations with possible inputs and outputs

"""We will construct a graph and then execute it"""


n1 = tf.constant(1)
n2 = tf.constant(2)
n3 = n1 + n2

g1 = tf.get_default_graph
print(g1)

g2 = tf.Graph()
print(g2)

#check and change the default graph
with g2.as_default():
    print(g2 is tf.get_default_graph)
print


with tf.Session() as sess:
    result = sess.run(n3)
    print(result)



