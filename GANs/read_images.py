import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from GANs.GAN import g_vars, generator,z,init

saver = tf.train.Saver(var_list=g_vars)
new_samples = []

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess,'./model/model_400.ckpt')
    for x in range(5):
        sample_z = np.random.uniform(-1, 1, size=(1, 100))
        gen_sample = sess.run(generator(z, reuse=True), feed_dict={z: sample_z})
        new_samples.append(gen_sample)

print("here")
plt.imshow(new_samples[0].reshape(28, 28), cmap='Greys')
plt.show()
input()
