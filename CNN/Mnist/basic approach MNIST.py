import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)

#PLACEHOLDERS :x for the image
x = tf.placeholder(tf.float32,shape=[None,784])


#VARIABLES : weights and bias
W = tf.Variable(tf.random_normal([784,10]))
b = tf.Variable(tf.random_normal([10]))


#CREATE GRAPH OPERATIONS
y = tf.matmul(x,W) + b


#LOSS FUNCTION
y_true = tf.placeholder(tf.float32,[None,10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y))


#OPTIMIZER
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)


#CREATE SESSION
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(2500):
        batch_x, batch_y = mnist.train.next_batch(batch_size=200)
        sess.run(train,feed_dict={x:batch_x,y_true:batch_y})

    #EVALUATE THE MODEL
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_true,1))
    #we have a list of true and falses, we convert it to floats
    acc = tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32))
    #this is a new graph
    print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels}))