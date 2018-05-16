import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

"""tf.estimator.LinearClassifier -> Constructs a neural network class model"""
"""tf.etimateor.DNNClassifier -> a neural network classification layer"""

"""1.Define a list of feature columns, 
2.create the model
3.create a data input function - numpy/pandas
4.call train, evaluate and predict methods on the estimator object"""

"""We create a large data set"""
x_data = np.linspace(0.0, 10.0, 1000000)
noise = np.random.randn(len(x_data))

# y = mx + b
# the predicted value, a random generated function
# we want to find our parameters to fit the line, aka smth close to 0.5 and 5
y_true = 0.5 * x_data + 5 + noise

x_df = pd.DataFrame(data=x_data, columns=['X Data'])
y_df = pd.DataFrame(data=y_true, columns=['Y'])

print(x_df.head())
print(y_df.head())
my_data = pd.concat([x_df, y_df], axis=1)
print(my_data.head())

my_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y')
plt.show()

# 10^points are kinda too many to train -> batch gradient descent
batch_size = 8

# now create the variables which we want to train
m1 = tf.Variable(0.81)
m2 = tf.Variable(0.32)
b = tf.Variable(0.17)

# now we need the placeholders
# x is feature, y is label
xph = tf.placeholder(tf.float32, [batch_size])
yph = tf.placeholder(tf.float32, [batch_size])

y_model = m1 * xph ** 2 + m2 * xph + b

# loss function
# error = tf.reduce_sum(tf.square(yph - y_model))
#
# # optimizer
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
# train = optimizer.minimize(error)
#
# init = tf.global_variables_initializer()

# with tf.Session() as sess:
#     sess.run(init)
#
#     batches = 1000
#     for i in range(batches):
#         # choose batch_size random numbers for the training
#         rand_ind = np.random.randint(len(x_data), size=batch_size)
#         feed = {xph: x_data[rand_ind], yph: y_true[rand_ind]}
#         sess.run(train, feed_dict=feed)
#         print(m1, m2, b)
#     print(sess.run([m1, m2, b]))
# # print(model_m2,model_m1)
# # print(model_m1,model_b)
#
# y_pred = x_data*model_m1 + model_b
# my_data.sample(250).plot(kind='scatter', x='X Data', y='Y')
# plt.plot(x_data,y_pred,'r')
# plt.show()


# 1 dimensional column
# aka a single feature
feat_cols = [tf.feature_column.numeric_column('x', shape=[1])]
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

#we split the data set
x_train,x_eval, y_train, y_eval = train_test_split(x_data,y_true,
                                                   test_size=0.3,random_state=101)


input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,
                                                batch_size=8,num_epochs=None,shuffle=True)
#we use this for evaluation against a test input function
train_input_funct = tf.estimator.inputs.numpy_input_fn({'x':x_train},
                            y_train,batch_size=8,num_epochs=1000,shuffle=False)

eval_input_funct = tf.estimator.inputs.numpy_input_fn({'x':x_eval},
                            y_eval,batch_size=8,num_epochs=1000,shuffle=False)

#first, train the estimator
estimator.train(input_fn = input_func,steps=1000)

train_metrics = estimator.evaluate(input_fn=train_input_funct,steps=1000)

eval_metrics = estimator.evaluate(input_fn=eval_input_funct,steps=1000)

#low loss on training and high on eval -> overfit
print("Training data metrics",train_metrics)
print("Evaluation data metrics",eval_metrics)

#never seen by the model
brand_new_data = np.linspace(0,10,10)
predict_input_funct = tf.estimator.inputs.numpy_input_fn({'x':brand_new_data},shuffle=False)
values = list(estimator.predict(input_fn=predict_input_funct))

predictions = []
for pred in values:
    predictions.append(pred['predictions'])
print(predictions)

my_data.sample(n=250).plot(kind ='scatter',x ='X Data',y = 'Y')
plt.plot(brand_new_data,predictions,'r*')
plt.show()