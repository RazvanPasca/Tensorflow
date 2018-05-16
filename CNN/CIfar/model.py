import tensorflow as tf

from CNN.CIfar.cifarhelper import *
from CNN.Mnist.CNNmnist import max_pool_2by2, conv_layer, conv2d, dense_layer

all_data = list(range(0, 7))
writer = tf.summary.FileWriter("demo/1")

for i, direc in zip(all_data, dirs):
    all_data[i] = unpickle(CIFAR_DIR + direc)

ch = CifarHelper(all_data)
ch.setup_test_images()
ch.setup_train_images()

# Placeholders
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, 10])
hold_prob = tf.placeholder(tf.float32)

conv1 = conv_layer(x, [4, 4, 3, 32], "conv1")
conv1_pool = max_pool_2by2(conv1, "pool1")

conv2 = conv_layer(conv1_pool, [4, 4, 32, 64], "conv2")
conv2_pool = max_pool_2by2(conv2, "pool2")

flatten = tf.reshape(conv2_pool, [-1, 8 * 8 * 64])
fc1 = tf.nn.relu(dense_layer(flatten, 1024))
fc1_dropout = tf.nn.dropout(fc1, keep_prob=hold_prob)

y_pred = dense_layer(fc1_dropout, 10)

with tf.name_scope("train"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
    train_function = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    tf.summary.scalar("cross_entropy", loss)

init = tf.global_variables_initializer()
training_steps = 1500

with tf.name_scope("accuracy"):
    matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

with tf.Session() as sess:
    sess.run(init)
    writer.add_graph(sess.graph)
    summaries = tf.summary.merge_all()
    for i in range(training_steps):
        x_batch, y_batch = ch.next_batch(batch_size=200)
        sess.run([train_function, loss], feed_dict={x: x_batch, y: y_batch, hold_prob: 0.5})
        if i % 5 == 0:
            s = sess.run(summaries, feed_dict={x: x_batch, y: y_batch, hold_prob: 1.0})
            writer.add_summary(s, i)
        if i % 100 is 0:
            acc = sess.run(accuracy, feed_dict={x: ch.test_images, y: ch.test_labels, hold_prob: 1.0})
            print(" Step {}, Accuracy {}".format(i, acc))
