import tensorflow as tf

# HELPER

# INIT WEIGHTS
def init_weights(shape,name ="weight_init"):
    with tf.name_scope(name):
        init_random_dist = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(init_random_dist)


# INIT BIAS
def init_bias(shape,name= "bias_init"):
    with tf.name_scope(name):
        init_bias_vals = tf.constant(0.1, shape=shape)
        return tf.Variable(init_bias_vals)


# CONV2D - wrapper
def conv2d(x, W):
    # x --> [batch,H,W,Channels]
    # W --> [filter_H,filter_W, Channels, nr_filters]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# POOLING -wrapper
def max_pool_2by2(x,name = "max_pool"):
    # x --> [batch,H,W,Channels]
    with tf.name_scope(name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name = name)


# CONVOLUTIONAL LAYER
def conv_layer(input_x, shape,name = "conv"):
    with tf.name_scope(name):
        W = init_weights(shape)
        b = init_bias([shape[3]])
        tf.summary.histogram("weights",W)
        tf.summary.histogram("biases",b)
        print(input_x.shape,W.shape)
        return tf.nn.relu(conv2d(input_x, W) + b,name = name)


# FULLY CONNECTED
def dense_layer(input_layer, size,name = "fc"):
    with tf.name_scope(name):
        input_size = int(input_layer.get_shape()[1])
        # shape[0] is nr of examples
        W = init_weights([input_size, size])
        b = init_bias([size])
        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        return tf.matmul(input_layer, W) + b

#KEEP YOUR DIMS IN MIND!!!!!

#
# #PLACEHOLDERS
# x = tf.placeholder(tf.float32,shape=[None,784])
# y_true = tf.placeholder(tf.float32,shape=[None,10])
#
#
# #LAYERS
# x_image = tf.reshape(x,[-1,28,28,1]) #greyscale, just 1 channel, want to unflatted the 784
# conv1 = conv_layer(x_image,shape = [5,5,1,32],name="conv1")
# #32 features for each 5x5 windows, with 1 input channel
# conv1_pool = max_pool_2by2(conv1,"pool1")
#
# conv2 = conv_layer(conv1_pool,shape = [5,5,32,64],name = "conv2")
# conv2_pool = max_pool_2by2(conv2,"pool2")
#
# conv2_flatten = tf.reshape(conv2_pool,[-1,7*7*64])
# dense_layer1 = tf.nn.relu(dense_layer(conv2_flatten,1024,name = "fc1"))
#
# #DROPOUT
# hold_prob = tf.placeholder(tf.float32)
# dense_layer1_dropout = tf.nn.dropout(dense_layer1,keep_prob=hold_prob)
# y_pred = dense_layer(dense_layer1_dropout,10,name = "output_fc")
#
#
# #LOSS FUNCTION
# with tf.name_scope("softmax"):
#     cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_true,logits=y_pred))
#     tf.summary.scalar("cross_entropy",cross_entropy)
#
#
# #OPTIMIZER
# with tf.name_scope("train"):
#     optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
#     train = optimizer.minimize(cross_entropy)
#
# init = tf.global_variables_initializer()
#
# steps = 1000
# writer = tf.summary.FileWriter("demo/4")
# with tf.Session() as sess:
#     sess.run(init)
#     merged_summaries = tf.summary.merge_all()
#     writer.add_graph(sess.graph)
#
#     for i in range(steps):
#         batch_x, batch_y = mnist.train.next_batch(32)
#         matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
#         acc = tf.reduce_mean(tf.cast(matches, tf.float32))
#         if i % 5 == 0:
#             s = sess.run(merged_summaries,feed_dict={x:batch_x,y_true:batch_y,hold_prob:1.0})
#             writer.add_summary(s,i)
#         _, loss = sess.run([train,acc], feed_dict={x:batch_x, y_true:batch_y, hold_prob:0.5})
#         with tf.name_scope("accuracy"):
#             if i % 100 is 0:
#             # ACCURACY
#             #     acc = sess.run(acc, feed_dict={x: mnist.test.images, y_true: mnist.test.labels, hold_prob: 1.0})
#                 tf.summary.scalar("accuracy", acc)
#                 print ("ON STEP {}, ACCURACY {}".format(i, loss))
