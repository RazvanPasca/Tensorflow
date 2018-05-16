import gym
import numpy as np
import tensorflow as tf

num_inputs = 4
num_hidden = 4
num_outputs = 1

learning_rate = 0.01

initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, num_inputs])

hidden_layer1 = tf.layers.dense(X, num_hidden, activation=tf.nn.relu, kernel_initializer=initializer)
logits = tf.layers.dense(hidden_layer1, num_outputs)
outputs = tf.nn.sigmoid(logits)
# hidden_layer2 = tf.layers.dense(hidden_layer1, num_hidden, activation=tf.nn.relu, kernel_initializer=initializer)
# output_layer = tf.layers.dense(hidden_layer2, num_outputs, activation=tf.nn.sigmoid, kernel_initializer=initializer)

probabilities = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(probabilities, num_samples=1)

y = 1.0 - tf.to_float(action)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y)
optimizer = tf.train.AdamOptimizer(learning_rate)

gradients_and_variables = optimizer.compute_gradients(cross_entropy)

gradients = []
gradient_placeholders = []
grads_and_vars_feed = []

for gradient, variable in gradients_and_variables:
    gradients.append(gradient)
    gradient_placeholder = tf.placeholder(tf.float32, shape=gradient.shape)
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append(((gradient_placeholder, variable)))

training_op = optimizer.apply_gradients(grads_and_vars_feed)
init = tf.global_variables_initializer()
saver = tf.train.Saver()


def helper_discount_rewards(rewards, discount_rate):
    '''
    Takes in rewards and applies discount rate
    '''
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards


def discount_and_normalize_rewards(all_rewards, discount_rate):
    '''
    Takes in all rewards, applies helper_discount function and then normalizes
    using mean and std.
    '''
    all_discounted_rewards = []
    for rewards in all_rewards:
        all_discounted_rewards.append(helper_discount_rewards(rewards, discount_rate))

    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]


env = gym.make('CartPole-v0')
num_rounds = 20
max_game_steps = 1000
epochs = 650
discount = 0.9

with tf.Session() as sess:
    sess.run(init)

    for iteration in range(epochs):
        print('On iteration: {}'.format(iteration))

        all_rewards = []
        all_gradients = []

        for game in range(num_rounds):
            current_rewards = []
            current_gradients = []

            obs = env.reset()

            for step in range(max_game_steps):
                action_val, gradients_val = sess.run([action, gradients], feed_dict={X: obs.reshape(1, num_inputs)})
                # we got back the action val and the gradients val
                # and perform it
                obs, reward, done, info = env.step(action_val[0][0])
                #get the current rewards and gradients
                current_rewards.append(reward)
                current_gradients.append(gradients_val)
                if done:
                    break

            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)

        #for each epoch
        all_rewards = discount_and_normalize_rewards(all_rewards,discount)
        feed_dict = {}

        #apply the scores calculated to the gradients
        # by taking the average of all gradients
        # gradient is reward*gradient for each game and step of particular game
        for var_index, gradient_placeholder in enumerate(gradient_placeholders):
            mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
                                      for game_index, rewards in enumerate(all_rewards)
                                      for step, reward in enumerate(rewards)], axis=0)
            feed_dict[gradient_placeholder] = mean_gradients

        sess.run(training_op, feed_dict=feed_dict)

    print('SAVING GRAPH AND SESSION')
    meta_graph_def = tf.train.export_meta_graph(filename='./models/my-650-step-model.meta')
    saver.save(sess, './models/my-650-step-model')

