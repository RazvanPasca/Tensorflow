import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../CNN/Mnist/MNIST_data/", one_hot=True)

writer = tf.summary.FileWriter("./tensorboard/NN")


def generator(z, alpha=0.01, reuse=None):
    with tf.variable_scope('gen', reuse=reuse):
        hidden1 = tf.layers.dense(inputs=z, units=128)
        hidden1 = tf.maximum(alpha * hidden1, hidden1)
        hidden2 = tf.layers.dense(inputs=hidden1, units=128)
        hidden2 = tf.maximum(alpha * hidden2, hidden2)
        output = tf.layers.dense(hidden2, units=784, activation=tf.nn.tanh)
        return output


def discriminator(X, alpha=0.01, reuse=None):
    with tf.variable_scope('dis', reuse=reuse):
        hidden1 = tf.layers.dense(inputs=X, units=128)
        hidden1 = tf.maximum(alpha * hidden1, hidden1)
        hidden2 = tf.layers.dense(inputs=hidden1, units=128)
        hidden2 = tf.maximum(alpha * hidden2, hidden2)
        logits = tf.layers.dense(hidden2, units=1)
        labels = tf.sigmoid(logits)
        return labels, logits


# Placeholders
real_images = tf.placeholder(tf.float32, shape=[None, 784])
z = tf.placeholder(tf.float32, shape=[None, 100])

G = generator(z)

D_output_real, D_logits_real = discriminator(real_images)
D_output_fake, D_logits_fake = discriminator(G, reuse=True)


# Losses
def loss_func(logits_in, labels_in):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in, labels=labels_in))


D_real_loss = loss_func(D_logits_real, tf.ones_like(D_logits_real) * 0.9)
D_fake_loss = loss_func(D_logits_fake, tf.zeros_like(D_logits_real))
# D loss is loss on real samples + loss on generated examples
D_loss = D_real_loss + D_fake_loss
# G loss is loss on fake labels which should have been 1
G_loss = loss_func(D_logits_fake, tf.ones_like(D_logits_fake))

# Optimizers
learning_rate = 0.001
tvars = tf.trainable_variables()

d_vars = [var for var in tvars if 'dis' in var.name]
g_vars = [var for var in tvars if 'gen' in var.name]

print([v.name for v in d_vars])
print([v.name for v in g_vars])

D_trainer = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=d_vars)
G_trainer = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=g_vars)

batch_size = 64
epochs = 500
init = tf.global_variables_initializer()
saver = tf.train.Saver(var_list=g_vars)
sample_z = np.random.uniform(-1, 1, size=(1, 100))

samples = []

# with tf.Session() as sess:
#     writer.add_graph(sess.graph)
#
#     sess.run(init)
#     for step in range(epochs):
#         num_batches = mnist.train.num_examples // batch_size
#         for b in range(num_batches):
#             # get images, reshape and rescale to pass to D
#             batch = mnist.train.next_batch(batch_size)
#             batch_images = batch[0].reshape((batch_size, 784))
#             batch_images = batch_images * 2 - 1
#
#             # generate noise for G
#             batch_z = np.random.uniform(-1, 1, size=(batch_size, 100))
#
#             # run optimizers
#             _ = sess.run(D_trainer, feed_dict={real_images: batch_images, z: batch_z})
#             _ = sess.run(G_trainer, feed_dict={z: batch_z})
#
#         print("On epoch {} out of {}".format(step + 1, epochs))
#
#         if step % 100:
#             gen_sample = sess.run(generator(z, reuse=True), feed_dict={z: sample_z})
#             samples.append(gen_sample)
#
#     for x in range(4):
#         sample_z = np.random.uniform(-1, 1, size=(1, 100))
#         gen_sample = sess.run(generator(z, reuse=True), feed_dict={z: sample_z})
#         samples.append(gen_sample)
#
#     save_path = saver.save(sess, "./models/500_epoch_model.ckpt")
#
#
# fig = plt.figure(figsize=(28, 28))
# for i in range(len(samples)):
#     fig.add_subplot(3, 3, i + 1)
#     plt.imshow(samples[i].reshape(28, 28), cmap='Greys')
# plt.show()

# thefile = open('test.txt', 'w')
# for item in samples:
#     thefile.write("%s\n" % item)

saver = tf.train.Saver(var_list=g_vars)
new_samples = []
fig = plt.figure(figsize=(28, 28))
with tf.Session() as sess:
    saver.restore(sess, './models/500_epoch_model.ckpt')

    for x in range(9):
        sample_z = np.random.uniform(-1, 1, size=(1, 100))
        gen_sample = sess.run(generator(z, reuse=True), feed_dict={z: sample_z})
        new_samples.append(gen_sample)

for i in range(len(new_samples)):
    fig.add_subplot(3, 3, i + 1)
    plt.imshow(new_samples[i].reshape(28, 28), cmap='Greys')

plt.show()
