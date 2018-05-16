import tensorflow as tf
print(tf.__version__)

hello = tf.constant("Hello")
world = tf.constant("World")
print(type(hello))


const = tf.constant(10)
fill__mat = tf.fill((4,4),10)
myzeros = tf.zeros((4,4))
myones = tf.ones(4,4)
myrandn = tf.random_normal((4,4),mean = 0, stddev = 1.0)
myrandu = tf.random_uniform((4,4),minval=0, maxval=1)

my_ops = [const,fill__mat,myzeros,myones,myrandn,myrandu]

a = tf.constant([[1,2],
                 [3,4]])
print(a.get_shape())

b = tf.constant([[10],
                 [100]])
result = tf.matmul(a, b)

#tensorflow operations that we run
with tf.Session() as sess:
    # for op in my_ops:
    #     print(sess.run(op),sep='\n')
    print(sess.run(result))
    


