import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

"""Predict from 1,2,3,4,5,6 the sequence 2,3,4,5,6,7 or even more difficult with 1 time ahead"""


class TimeSeriesData():
    def __init__(self, num_points, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
        self.num_points = num_points
        self.resolution = (xmax - xmin) / num_points
        self.x_data = np.linspace(xmin, xmax, num_points)
        self.y_true = np.sin(self.x_data)

    def ret_true(self, x_series):
        return np.sin(x_series)

    def next_batch(self, batch_size, nr_steps, return_batch_ts=False):
        """Need to grab a random starting point for each batch,
            Convert it to be on time series
            Create a batch time series on the X axis
            Create the Y data for the time series X axis + format it"""
        random_start = np.random.rand(batch_size, 1)
        ts_start = random_start * (self.xmax - self.xmin - (nr_steps * self.resolution))
        batch_ts = ts_start + np.arange(0.0, nr_steps + 1) * self.resolution
        y_batch = np.sin(batch_ts)
        if return_batch_ts:
            return y_batch[:, :-1].reshape(-1, nr_steps, 1), y_batch[:, 1:].reshape(-1, nr_steps, 1), batch_ts
        else:
            # this are the 2 batches, one being shifted into the future with 1 step
            return y_batch[:, :-1].reshape(-1, nr_steps, 1), y_batch[:, 1:].reshape(-1, nr_steps, 1)


ts_data = TimeSeriesData(250, 0, 10)

nr_steps = 32
# print(y1)
# plt.show()

train_inst = np.linspace(5, 5 + ts_data.resolution * (nr_steps + 1), nr_steps + 1)

# THE MODEL
tf.reset_default_graph()

nr_inputs = 1
nr_neurons = 100
nr_outputs = 1
learning_rate = 0.001
nr_epochs = 2000
batch_size = 1

X = tf.placeholder(tf.float32, [None, nr_steps, nr_inputs])
Y = tf.placeholder(tf.float32, [None, nr_steps, nr_outputs])

# The cell layer
cell = tf.contrib.rnn.BasicRNNCell(num_units=nr_neurons, activation=tf.nn.relu)
cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size=nr_outputs)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
loss = tf.reduce_mean(tf.square(outputs - Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
predictions = []

with tf.Session() as sess:
    sess.run(init)
    for iter in range(nr_epochs):
        X_batch, Y_batch = ts_data.next_batch(batch_size, nr_steps)
        sess.run(train, feed_dict={X: X_batch, Y: Y_batch})

        if iter % 100 == 0:
            mse = loss.eval(feed_dict={X: X_batch, Y: Y_batch})
            print(iter, "\tMSE", mse)
            X_new = np.sin(np.array(train_inst[:-1].reshape(-1, nr_steps, nr_inputs)))
            y_pred = sess.run(outputs, feed_dict={X: X_new})
            predictions.append(y_pred)

    saver.save(sess, "./rnn_time_series_model_from_train")

y0 = predictions[0]
y1000 = predictions[10]
y2000 = predictions[-1]
plt.plot(train_inst[:-1], np.sin(train_inst[:-1]), "bo", markersize=16, alpha=0.5, label='Training Inst')

plt.plot(train_inst[1:], np.sin(train_inst[1:]), "ko", markersize=10, label='target')

plt.plot(train_inst[1:],y0[0,:,0],'r.',markersize = 10, label = 'Predictions')

plt.plot(train_inst[1:],y1000[0,:,0],'g.',markersize = 10, label = 'Predictions')

plt.plot(train_inst[1:],y2000[0,:,0],'y.',markersize = 10, label = 'Predictions')

plt.legend()
plt.show()