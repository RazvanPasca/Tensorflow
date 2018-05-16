import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(101)
tf.set_random_seed(101)



rand_a = np.random.uniform(0,100,(5,5))
# print(rand_a)
rand_b = np.random.uniform(0,100,(5,1))
# print(rand_b)


#we multiply features by weights
#aka (y,10)*(10,3) because we have 3 neurons
n_features = 10
n_dense_neurons = 3
x = tf.placeholder(tf.float32,(None,n_features))
w = tf.Variable(tf.random_normal([n_features,n_dense_neurons]))
b = tf.Variable(tf.ones([n_dense_neurons]))
xW = tf.matmul(x,w)
z = tf.add(xW,b)
label = tf.sigmoid(z)

"""Now comes a linear regression example"""
#10 points between 0, 10 + some noise
x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,-1.5,10)

y_label = np.linspace(0,10,10) +np.random.uniform(-1.5,1.5,10)


plt.plot(x_data,y_label,'*')
plt.show()
#we get 2 random numbers, which will be optimized
m = tf.Variable(0.44)
b = tf.Variable(0.87)

error = 0

#cost function
for x,y in zip(x_data,y_label):
    y_hat = m*x + b
    error += (y-y_hat)**2

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    #how many steps i'm going to perform
    training_steps = 10
    for i in range(training_steps):
        sess.run(train)
    final_slope,final_intercept = sess.run([m,b])

x_test = np.linspace(-1,11,10)
# we use the line predicted by the model
y_pred_plot = final_slope*x_test + final_intercept
plt.plot(x_test,y_pred_plot,'r')
plt.plot(x_data,y_label,'*')
plt.show()