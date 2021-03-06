import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow import nn, layers
from tensorflow.contrib import layers as clayers

mnist = input_data.read_data_sets("../CNN/Mnist/MNIST_data/", one_hot=True)

writer = tf.summary.FileWriter("./tensorboard/CNN")

def default_conv2d(inputs, filters):
    return layers.conv2d(
        inputs,
        filters=filters,
        kernel_size=4,
        strides=(2, 2),
        padding='same',
        data_format='channels_last',
        use_bias=False,
    )


def default_conv2d_transpose(inputs, filters):
    return layers.conv2d_transpose(
        inputs,
        filters=filters,
        kernel_size=4,
        strides=(2, 2),
        padding='same',
        data_format='channels_last',
        use_bias=False,
    )


def noise(n_rows, n_cols):
    return np.random.normal(size=(n_rows, n_cols))


def generator(z, reuse=tf.AUTO_REUSE):
    with tf.variable_scope('gen', reuse=reuse):
        with tf.variable_scope("linear"):
            linear = clayers.fully_connected(z, 128 * 2 * 2)

        with tf.variable_scope("conv1_transp"):
            # Reshape as 4x4 images
            conv1 = tf.reshape(linear, (-1, 2, 2, 128))
            conv1 = default_conv2d_transpose(conv1, 64)
            conv1 = nn.relu(conv1)

        with tf.variable_scope("conv2_transp"):
            conv2 = default_conv2d_transpose(conv1, 32)
            conv2 = nn.relu(conv2)

        with tf.variable_scope("conv3_transp"):
            conv3 = default_conv2d_transpose(conv2, 16)
            conv3 = nn.relu(conv3)

        with tf.variable_scope("conv4_transp"):
            conv4 = default_conv2d_transpose(conv3, 1)

        with tf.variable_scope("out"):
            out = tf.tanh(conv4)
        return out


def discriminator(x, alpha=0.01, reuse=tf.AUTO_REUSE):
    with tf.variable_scope('dis', reuse=reuse):
        with tf.variable_scope("conv1"):
            conv1 = default_conv2d(x, 16)
            conv1 = nn.relu(conv1)

        with tf.variable_scope("conv2"):
            conv2 = default_conv2d(conv1, 32)
            conv2 = nn.relu(conv2)

        with tf.variable_scope("conv3"):
            conv3 = default_conv2d(conv2, 64)
            conv3 = nn.relu(conv3)

        with tf.variable_scope("conv4"):
            conv4 = default_conv2d(conv3, 128)
            conv4 = nn.relu(conv4)

        with tf.variable_scope("linear"):
            linear = clayers.flatten(conv4)
            linear = clayers.fully_connected(linear, 1)

        with tf.variable_scope("out"):
            out = nn.sigmoid(linear)
    return out


# Placeholders
real_images = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
z = tf.placeholder(tf.float32, shape=[None, 100])

G = generator(z)

D_output_real = discriminator(real_images)
D_output_fake = discriminator(G)


# Losses
def loss_func(logits_in, labels_in):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in, labels=labels_in))


D_real_loss = loss_func(D_output_real, tf.ones_like(D_output_real) * 0.9)
D_fake_loss = loss_func(D_output_fake, tf.zeros_like(D_output_fake))
# D loss is loss on real samples + loss on generated examples
D_loss = D_real_loss + D_fake_loss
# G loss is loss on fake labels which should have been 1
G_loss = loss_func(D_output_fake, tf.ones_like(D_output_fake))

# Optimizers
learning_rate = 0.001
tvars = tf.trainable_variables()

d_vars = [var for var in tvars if 'dis' in var.name]
g_vars = [var for var in tvars if 'gen' in var.name]

D_trainer = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=d_vars)
G_trainer = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=g_vars)

batch_size = 32
epochs = 200
init = tf.global_variables_initializer()
saver = tf.train.Saver(var_list=g_vars)
sample_z = np.random.uniform(-1, 1, size=(1, 100))

samples = []

# print("starting the session")
#
# with tf.Session() as sess:
#     sess.run(init)
#     writer.add_graph(sess.graph)
#
#     for step in range(epochs):
#         start = time.perf_counter()
#         num_batches = mnist.train.num_examples // batch_size
#
#         for b in range(num_batches):
#             # get images, reshape and rescale to pass to D
#             batch = mnist.train.next_batch(batch_size)
#             batch_images = batch[0].reshape(-1, 28, 28, 1)
#             batch_images = batch_images * 2 - 1
#
#             # generate noise for G
#             batch_z = np.random.uniform(-1, 1, size=(batch_size, 100))
#
#             # run optimizers
#             _ = sess.run(D_trainer, feed_dict={real_images: batch_images, z: batch_z})
#             _ = sess.run(G_trainer, feed_dict={z: batch_z})
#
#         elapsed = time.perf_counter() - start
#         print('Elapsed %.3f seconds.' % elapsed, end=" ")
#         print("On epoch {} out of {}".format(step + 1, epochs))
#
#         if step % 100:
#             gen_sample = sess.run(generator(z), feed_dict={z: sample_z})
#             samples.append(gen_sample)
#     save_path = saver.save(sess, "./models/200_epoch_model_DCGAN.ckpt")
#
# plt.imshow(samples[0].reshape(28, 28), cmap='Greys')
# plt.show()
#
# thefile = open('test_DCGAN.txt', 'w')
# for item in samples:
#     thefile.write("%s\n" % item)

saver = tf.train.Saver(var_list=g_vars)
new_samples = []
fig = plt.figure(figsize=(32,32))

with tf.Session() as sess:
    saver.restore(sess, './models/200_epoch_model_DCGAN.ckpt')

    for x in range(9):
        sample_z = np.random.uniform(-1, 1, size=(1, 100))
        gen_sample = sess.run(generator(z), feed_dict={z: sample_z})
        new_samples.append(gen_sample)

for i in range(len(new_samples)):
    fig.add_subplot(3,3,i+1)
    plt.imshow(new_samples[i].reshape(32, 32), cmap='Greys')
plt.show()

