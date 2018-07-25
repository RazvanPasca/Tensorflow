import os
import tensorflow as tf
import numpy as np
from random import shuffle
from matplotlib import pyplot as plt

# HELPER
from CNN.image_gen import read_image, resize


class Dataset:
    def __init__(self, images_path):
        self.images, self.labels = self.create_dataset(images_path)
        self.train_X = self.images[0:1984]
        self.train_Y = self.labels[0:1984]
        self.test_X = self.images[1984:]
        self.test_Y = self.labels[1984:]
        self.size = self.train_X.shape[0]
        self.batches_generated = 0

    def create_dataset(self, images_path):
        images = []
        labels = []
        for dirpath, dirs, files in os.walk(images_path):
            for filename in files:
                fname = os.path.join(dirpath, filename)
                inimg = read_image(fname)
                if inimg.shape == (100,100,3):
                    images.append(inimg)

                labels.append([1, 0] if 'Apple' in dirpath else [0, 1])
        images = np.array(images)
        labels = np.array(labels)
        print(images.shape)
        print(labels.shape)
        return self.shuffle_dataset(images, labels)

    def shuffle_dataset(self, images=None, labels=None):
        if images is None:
            nr_examples = self.train_X.shape[0]
            shuffle_seed = list(np.random.permutation(nr_examples))
            self.train_X, self.train_Y = self.train_X[shuffle_seed], self.train_Y[shuffle_seed].reshape(nr_examples, 2)
        else:
            nr_examples = images.shape[0]
            shuffle_seed = list(np.random.permutation(nr_examples))
            return images[shuffle_seed], labels[shuffle_seed].reshape(nr_examples, 2)

    def next_batch(self, batch_size):
        nr_batches = np.math.floor(self.size / batch_size)
        if self.batches_generated < nr_batches:
            images_batch = self.train_X[self.batches_generated * batch_size:(self.batches_generated + 1) * batch_size]
            labels_batch = self.train_Y[self.batches_generated * batch_size:(self.batches_generated + 1) * batch_size]
            self.batches_generated += 1
            return images_batch, labels_batch
        else:
            self.batches_generated = 0
            self.shuffle_dataset()
            return self.next_batch(batch_size)
        # start_pos = np.random.randint(0, self.size - batch_size)
        # return self.images[start_pos:start_pos + batch_size], self.labels[start_pos:start_pos + batch_size]


def get_label(prediction):
    return "Apple" if prediction == 0 else "Banana"


# INIT WEIGHTS
def init_weights(shape, name="weight_init"):
    with tf.name_scope(name):
        init_random_dist = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(init_random_dist)


# INIT BIAS
def init_bias(shape, name="bias_init"):
    with tf.name_scope(name):
        init_bias_vals = tf.constant(0.1, shape=shape)
        return tf.Variable(init_bias_vals)


# CONV2D - wrapper
def conv2d(x, W):
    # x --> [batch,H,W,Channels]
    # W --> [filter_H,filter_W, Channels, nr_filters]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# POOLING -wrapper
def max_pool_2by2(x, name="max_pool"):
    # x --> [batch,H,W,Channels]
    with tf.name_scope(name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


# CONVOLUTIONAL LAYER
def conv_layer(input_x, shape, name="conv"):
    with tf.name_scope(name):
        W = init_weights(shape)
        b = init_bias([shape[3]])
        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        print(input_x.shape, W.shape)
        return tf.nn.relu(conv2d(input_x, W) + b, name=name)


# FULLY CONNECTED
def dense_layer(input_layer, size, name="fc"):
    with tf.name_scope(name):
        input_size = int(input_layer.get_shape()[1])
        # shape[0] is nr of examples
        W = init_weights([input_size, size])
        b = init_bias([size])
        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        return tf.matmul(input_layer, W) + b


# KEEP YOUR DIMS IN MIND!!!!!

dataset = Dataset("fruits")

# PLACEHOLDERS
x = tf.placeholder(tf.float32, shape=[None, 100, 100, 3])
y_true = tf.placeholder(tf.float32, shape=[None, 2])

# LAYERS
conv1 = conv_layer(x, shape=[5, 5, 3, 32], name="conv1")
# 32 features for each 5x5 windows, with 1 input channel
conv1_pool = max_pool_2by2(conv1, "pool1")

conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64], name="conv2")
conv2_pool = max_pool_2by2(conv2, "pool2")

conv2_flatten = tf.reshape(conv2_pool, [-1, 25 * 25 * 64])
dense_layer1 = tf.nn.relu(dense_layer(conv2_flatten, 1024, name="fc1"))

# DROPOUT
hold_prob = tf.placeholder(tf.float32)
dense_layer1_dropout = tf.nn.dropout(dense_layer1, keep_prob=hold_prob)
y_pred = dense_layer(dense_layer1_dropout, 2, name="output_fc")

# LOSS FUNCTION

with tf.name_scope("softmax"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
    tf.summary.scalar("cross_entropy", cross_entropy)

# OPTIMIZER
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

steps = 300
writer = tf.summary.FileWriter("demo/fruits")
saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init)
#
#     writer.add_graph(sess.graph)
#
#     for i in range(steps):
#         batch_x, batch_y = dataset.next_batch(32)
#         with tf.name_scope("accurracy"):
#             matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
#             acc = tf.reduce_mean(tf.cast(matches, tf.float32))
#             tf.summary.scalar("accuracy", acc)
#         merged_summaries = tf.summary.merge_all()
#
#         _, acc_train = sess.run([train, acc], feed_dict={x: batch_x, y_true: batch_y, hold_prob: 0.5})
#         if i % 5 == 0:
#             s = sess.run(merged_summaries, feed_dict={x: batch_x, y_true: batch_y, hold_prob: 1.0})
#             writer.add_summary(s, i)
#             with tf.name_scope("accuracy_test"):
#                 # ACCURACY
#                 test_X, test_y = dataset.test_X, dataset.test_Y
#                 acc = sess.run(acc, feed_dict={x: test_X, y_true: test_y, hold_prob: 1.0})
#                 print("ON STEP {}, ACCURACY train {} ACCURACY test {}".format(i, acc_train, acc))
#
#     save_path = saver.save(sess, "./fruit_models/model_2CNN_layers_green.ckpt")
#     print("Model saved in path: %s" % save_path)

test_image = read_image("./original.jpeg")
plt.imshow(test_image)
plt.show()
test_image = resize(test_image, 100).reshape(-1, 100, 100, 3)

with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "./fruit_models/model_2CNN_layers_green.ckpt")
    prediction = sess.run(tf.argmax(y_pred,1), feed_dict={x: test_image, hold_prob: 1.0})
    print(get_label(prediction))
