
import tensorflow as tf
import csv
import random
import numpy as np
import datetime
import readDatasetNNInSequence as load
import sys

def train(dataset):
    learning_rate = 0.001
    batch_size = 100
    epoch = 10000
    keep_prob = 0.7
    beta = 0.01

    # x_input = np.array(x_data)
    # y_label = np.array(y_data)

    # print(y_label)

    weights = {
        "W1": tf.Variable(tf.random_normal([122, 64])),
        "W2": tf.Variable(tf.random_normal([64, 64])),
        "W3": tf.Variable(tf.random_normal([64, 64])),
        "W4": tf.Variable(tf.random_normal([64, 64])),
        "W5": tf.Variable(tf.random_normal([64, 2])),
    }

    biases = {
        "b1": tf.Variable(tf.random_normal([64])),
        "b2": tf.Variable(tf.random_normal([64])),
        "b3": tf.Variable(tf.random_normal([64])),
        "b4": tf.Variable(tf.random_normal([64])),
        "b5": tf.Variable(tf.random_normal([2])),
    }

    X = tf.placeholder(tf.float32, shape = [None, 122])
    Y = tf.placeholder(tf.float32, shape = [None, 2])

    # ================ Batch Normalisation ====================
    input_layer = tf.nn.sigmoid(tf.add(tf.matmul(X, weights["W1"]), biases["b1"]))
    h1 = tf.nn.sigmoid(tf.add(tf.matmul(input_layer, weights["W2"]), biases["b2"]))
    h1 = tf.contrib.layers.batch_norm(h1, center = True, scale = True, is_training = True)

    h2 = tf.nn.relu(tf.add(tf.matmul(h1, weights["W3"]), biases["b3"]))
    h2 = tf.contrib.layers.batch_norm(h2, center = True, scale = True, is_training = True)

    h3 = tf.nn.sigmoid(tf.add(tf.matmul(h2, weights["W4"]), biases["b4"]))

    # ================ Dropout ====================
    h3 = tf.nn.dropout(h3, keep_prob)
    output_layer = tf.add(tf.matmul(h3, weights["W5"]), biases["b5"])

    predicted = tf.nn.softmax(output_layer)

    # ================ L2 Regulariser ==================
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output_layer, labels = Y))
    regulariser = tf.nn.l2_loss(weights['W5'])
                  # + tf.nn.l2_loss(weights['W4']) \
                  # + tf.nn.l2_loss(weights['W3']) + tf.nn.l2_loss(weights['W2']) \
                  # + tf.nn.l2_loss(weights['W1'])
    loss = tf.reduce_mean(cost + beta * regulariser)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        train_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    correction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
    acc = tf.reduce_mean(tf.cast(correction, tf.float32))

    saver = tf.train.Saver()

    start_time = datetime.datetime.now()
    print(start_time)

    with tf.Session() as sess:
        sess.run(init)

        for step in range(epoch):
            x_input, y_label = dataset.next_batch(batch_size)

            sess.run(train_op, feed_dict = {X: x_input, Y: y_label})

            if step % 1000 == 0:
                print(step, sess.run(cost, feed_dict = {X: x_input, Y: y_label}),
                      "W5: ", sess.run(weights["W5"]))

        saver.save(sess, "./FCN_result_inSequence/us_bank_training.ckpt")

        # extract only fraud data
        x_train_input, y_train_input = dataset.get_train_data()
        x_test_input, y_test_input = dataset.get_test_data()

        num_train = len(x_train_input)
        num_test = len(x_test_input)
        print("num of train: " + str(num_train))
        print("num of test: " + str(num_test))

        _x = np.array(x_train_input)
        _y = np.array(y_train_input)

        predicted = sess.run(tf.floor(predicted + 0.5), feed_dict = {X: _x})

        true = 0
        false = 0
        total = len(predicted)
        for i in range(len(predicted)):
            if np.array_equal(predicted[i], [0, 1]):
                true += 1
            else:
                false += 1

        print("Accuracy: ", acc.eval({X: _x, Y: _y}))
        # =========================================================
        print(true)
        print(false) # non fraud


if __name__ == "__main__":
    dataset = load.readDatasetNNInSequence()
    # dataset.split_dataset() # make k fold dataset
    dataset.preprocessing()

    s_time = datetime.datetime.now()
    train(dataset)
    e_time = datetime.datetime.now()
    print("Training time: " + str(e_time - s_time))





