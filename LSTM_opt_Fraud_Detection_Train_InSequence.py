
"""
    Written by Taejoon Kim
    20. Jan. 2018. Saturday
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import readDatasetLSTMInSequence as load
import datetime


def train():
    dataset = load.readDatasetLSTMInSequence()
    # dataset.split_dataset()
    dataset.preprocessing()

    # Training Parameters
    learning_rate = 0.001
    training_steps = 50000
    display_step = 1000
    batch_size = 100

    # Network Parameters
    num_input = 122 # Kaggle credit card fraud dataset
    timesteps = 1 # timesteps
    num_hidden = 10 # hidden layer num of features
    num_classes = 2 # datasets classes (0 or 1)
    beta = 0.01
    keep_prob = 0.75

    # tf Graph input
    X = tf.placeholder(tf.float32, [None, timesteps, num_input])
    Y = tf.placeholder(tf.float32, [None, num_classes])

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
    }

    biases = {
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    def RNN(x, weights, biases):

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, timesteps, n_input)
        # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
        x = tf.unstack(x, timesteps, 1)
        # x = x.reshape((batch_size, timesteps, num_input))

        # Define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias = 1)

        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype = tf.float32)
        # print(outputs)

        drop_out = tf.nn.dropout(outputs, keep_prob)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(drop_out[-1], weights['out']) + biases['out']

    logits = RNN(X, weights, biases)

    # Batch Normalisation
    logits = tf.contrib.layers.batch_norm(logits, center = True, scale = True, is_training = True)

    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer and applying L2 regularisation with beta 0.01
    # ============================== L2 Regulariser ================================
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y))
    regulariser = tf.nn.l2_loss(weights['out'])
    loss = tf.reduce_mean(cost + beta * regulariser)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        train_op = optimizer.minimize(loss)

    # Evaluate model (with test logits, for dropout to be disabled)
    correction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    acc = tf.reduce_mean(tf.cast(correction, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)

        for step in range(1, training_steps + 1):
            # Run optimisation op (backpropagation)
            train_x, train_y = dataset.next_batch(batch_size)

            sess.run(train_op, feed_dict={X: train_x, Y: train_y})

            if step % display_step == 0 or step == 1:
                loss, accuracy = sess.run([cost, acc], feed_dict = {X: train_x, Y: train_y})
                print("Step: " + str(step) + " Loss: " + "{:.4f}".format(loss) + " Accuracy: " + "{:.2f}".format(accuracy))

                """
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={X: train_x, Y: train_y})
                print("Step " + str(step) + ", Loss = "
                      + "{:.4f}".format(loss) + ", Training Accuracy = "
                      + "{:.3f}".format(acc))
                """

        print("Optimisation Finished!")
        saver.save(sess, "./LSTM_result_inSequence/result_trained_LSTM_inSequence.ckpt")

        # train_x, train_y, test_x, test_y = load.get_data()

        test_x, test_y = dataset.get_test_data()
        train_x, train_y = dataset.get_train_data()

        # y_predicted = sess.run(tf.floor(prediction + 0.5), feed_dict = {X: test_x})
        y_predicted = sess.run(tf.floor(prediction + 0.5), feed_dict = {X: test_x})

        total = len(test_y)

        true = 0
        false = 0
        for i in range(len(y_predicted)):
            if np.array_equal(y_predicted[i], test_y[i]):
                true += 1
            else:
                false += 1

        fraud = 0
        non_fraud = 0
        for i in range(len(y_predicted)):
            if np.array_equal(y_predicted[i], [0, 1]):
                fraud += 1
            else:
                non_fraud += 1

        print("========================================")
        accuracy = (true / total) * 100
        print("accuracy: %.2f" % accuracy + "%")
        print(fraud)
        print(non_fraud)


if __name__ == "__main__":
    start = datetime.datetime.now()

    train()

    end = datetime.datetime.now()
    print("Training time: " + str(end - start))



