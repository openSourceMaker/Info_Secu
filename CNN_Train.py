
import tensorflow as tf
import numpy as np
import csv
import readDatasetNNInSequence as load
import datetime
import sys

X = tf.placeholder(tf.float32, shape = [None, 122])
Y = tf.placeholder(tf.float32, shape = [None, 2])

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    # tf.nn.conv2d(input, filter, strides, padding, ...)
    # Computes a 2-D convolution given 4-D input and filter tensors.
    # input is equal to image
    x = tf.nn.conv2d(x, W, strides=[1, strides, 61, 1], padding='SAME')

    # tf.nn.bias_add(value, bias)
    # add bias to value
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, 1, k, 1], strides=[1, 1, k, 1], padding='SAME')


# Create model
def conv_net(x, weights, biases):
    # Reshape input picture
    # x = tf.reshape(x, shape=[-1, 1, 4, 1])
    x = tf.reshape(x, shape=[-1, 1, 122, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['c_w1'], biases['c_b1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, 2)
    print(conv1)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    # tf.reshape(tensor, shape, ...)
    fc1 = tf.reshape(conv1, [-1, weights['w1'].get_shape().as_list()[0]])
    print(fc1)
    fc1 = tf.add(tf.matmul(fc1, weights['w1']), biases['b1'])
    # fc1 = tf.add(tf.matmul(conv1, weights['w1']), biases['b1'])
    fc1 = tf.nn.relu(fc1)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    # Output, class prediction
    return out


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
        # layer 1 x 28, 1 input, 8 outputs
        'c_w1': tf.Variable(tf.random_normal([1, 122, 1, 20])),
        'w1': tf.Variable(tf.random_normal([20, 10])),
        'out': tf.Variable(tf.random_normal([10, 2]))
    }


    biases = {
        'c_b1': tf.Variable(tf.random_normal([20])),
        'b1': tf.Variable(tf.random_normal([10])),
        'out': tf.Variable(tf.random_normal([2]))
    }

    # # ================ Batch Normalisation ====================
    # input_layer = tf.nn.sigmoid(tf.add(tf.matmul(X, weights["W1"]), biases["b1"]))
    # h1 = tf.nn.sigmoid(tf.add(tf.matmul(input_layer, weights["W2"]), biases["b2"]))
    # h1 = tf.contrib.layers.batch_norm(h1, center = True, scale = True, is_training = True)
    #
    # h2 = tf.nn.relu(tf.add(tf.matmul(h1, weights["W3"]), biases["b3"]))
    # h2 = tf.contrib.layers.batch_norm(h2, center = True, scale = True, is_training = True)
    #
    # h3 = tf.nn.sigmoid(tf.add(tf.matmul(h2, weights["W4"]), biases["b4"]))
    #
    # # ================ Dropout ====================
    # h3 = tf.nn.dropout(h3, keep_prob)
    # output_layer = tf.add(tf.matmul(h3, weights["W5"]), biases["b5"])

    pred = conv_net(X, weights, biases)

    predicted = tf.nn.softmax(pred)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
    optimiser = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    # # ================ L2 Regulariser ==================
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output_layer, labels = Y))
    # regulariser = tf.nn.l2_loss(weights['W5'])
    # # + tf.nn.l2_loss(weights['W4']) \
    # # + tf.nn.l2_loss(weights['W3']) + tf.nn.l2_loss(weights['W2']) \
    # # + tf.nn.l2_loss(weights['W1'])
    # loss = tf.reduce_mean(cost + beta * regulariser)
    #
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    #     optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    #     train_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    correction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    acc = tf.reduce_mean(tf.cast(correction, tf.float32))

    saver = tf.train.Saver()

    start_time = datetime.datetime.now()
    print(start_time)

    with tf.Session() as sess:
        sess.run(init)

        for step in range(epoch):
            x_input, y_label = dataset.next_batch(batch_size)

            sess.run(optimiser, feed_dict = {X: x_input, Y: y_label})

            if step % 1000 == 0:
                loss, accuracy = sess.run([cost, acc], feed_dict = {X: x_input, Y: y_label})

                print("Step " + str(step) + ", Loss= " +
                      "{:.5f}".format(loss) + ", Accuracy= " +
                      "{:.5f}".format(accuracy))

        saver.save(sess, "./CNN_result_inSequence/us_bank_training.ckpt")

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
        print("Number of fraud: " + str(true))
        print("Number of normal: " + str(false)) # non fraud


if __name__ == "__main__":
    dataset = load.readDatasetNNInSequence()
    # dataset.split_dataset() # make k fold dataset
    dataset.preprocessing()

    s_time = datetime.datetime.now()
    train(dataset)
    e_time = datetime.datetime.now()
    print("Training time: " + str(e_time - s_time))




