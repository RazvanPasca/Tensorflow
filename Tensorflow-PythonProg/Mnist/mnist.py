import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

L1_nodes = 100
L2_nodes = 200
L3_nodes = 100

n_classes = 10
batch_size = 128
nr_examples = mnist.train.num_examples

epochs = 100
learning_rate = 0.001

X = tf.placeholder(dtype=tf.float32, shape=[None, 28 * 28])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 10])


def create_model(X):
    w1 = tf.Variable(tf.random_normal([28 * 28, L1_nodes]))
    w2 = tf.Variable(tf.random_normal([L1_nodes, L2_nodes]))
    w3 = tf.Variable(tf.random_normal([L2_nodes, L3_nodes]))
    output_w = tf.Variable(tf.random_normal([L3_nodes, n_classes]))

    L1 = tf.matmul(X, w1)
    L1 = tf.nn.relu(L1)
    L2 = tf.matmul(L1, w2)
    L2 = tf.nn.relu(L2)
    L3 = tf.matmul(L2, w3)
    L3 = tf.nn.relu(L3)
    output_layer = tf.matmul(L3, output_w)
    return output_layer


def train_network(X, learining_rate, epochs):
    prediction = create_model(X)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learining_rate).minimize(cost)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        for epoch in range(epochs):
            loss = 0
            for i in range(int(nr_examples / batch_size)):
                x, y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={X: x, Y: y})
                loss += c
            print('Epoch', epoch, 'out of', epochs, 'loss:', loss)
            
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
            acc = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy test:', acc.eval({X: mnist.test.images, Y: mnist.test.labels}))


train_network(X, learning_rate, epochs)
