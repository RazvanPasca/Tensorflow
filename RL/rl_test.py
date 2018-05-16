import gym
import tensorflow as tf
import numpy as np

num_inputs = 4
num_hidden = 4
num_outputs = 1

initializer = tf.contrib.layers.variance_scaling_initializer()
X = tf.placeholder(tf.float32, shape=[None, num_inputs])
Y = tf.placeholder(tf.float32, shape = [None,num_outputs])

hidden_layer1 = tf.layers.dense(X, num_hidden, activation=tf.nn.relu, kernel_initializer=initializer)
hidden_layer2 = tf.layers.dense(hidden_layer1, num_hidden, activation=tf.nn.relu, kernel_initializer=initializer)
output_layer = tf.layers.dense(hidden_layer2, num_outputs, activation=tf.nn.sigmoid, kernel_initializer=initializer)

probabilities = tf.concat(axis=1, values=[output_layer, 1 - output_layer])
action = tf.multinomial(probabilities, num_samples=1)

error = (0-output_layer)**2

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

episodes = 50
step_limit = 500
avg_steps = []

env = gym.make('CartPole-v0')

with tf.Session() as sess:
    init.run()
    for episode in range(episodes):
        obs = env.reset()
        for step in range(step_limit):
            action_val = action.eval(feed_dict={X: obs.reshape(1, num_inputs)})
            obs, reward, done, info = env.step(action_val[0][0])
            sess.run(train,feed_dict={X:obs.reshape(1,num_inputs)})
            if done:
                avg_steps.append(step)
                print("Done after {} steps".format(step))
                break

print("after {} episodes, average step per game was {}".format(episodes,np.mean(avg_steps)))
env.close()
