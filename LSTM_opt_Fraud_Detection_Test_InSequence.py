
"""
    Written by Taejoon Kim
    20. Jan. 2018. Saturday
"""

import csv
import itertools

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, roc_curve, auc
from tensorflow.contrib import rnn

import readDatasetLSTMInSequence as load


def test():
    dataset = load.readDatasetLSTMInSequence()
    # dataset.split_dataset()
    dataset.preprocessing()

    # Network Parameters
    num_input = 122 # Kaggle credit card fraud dataset
    timesteps = 1 # timesteps
    num_hidden = 10 # hidden layer num of features
    num_classes = 2 # datasets classes (0 or 1)

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

        # Define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias = 1.0)

        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
        # print(outputs)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

    logits = RNN(X, weights, biases)

    logits = tf.contrib.layers.batch_norm(logits, center = True, scale = True, is_training = False)

    # prediction = tf.nn.sigmoid(logits)
    prediction = tf.nn.softmax(logits)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    saver = tf.train.Saver()

    # Start training
    with tf.Session() as sess:
        # Run the initializer

        saver.restore(sess, "./LSTM_result_inSequence/result_trained_LSTM_inSequence.ckpt")

        test_x, test_y = dataset.get_test_data()
        y_predicted = sess.run(tf.floor(prediction + 0.5), feed_dict = {X: test_x})

        acc = accuracy.eval({X: test_x, Y: test_y})
        print("Testing Accuracy: %.2f" % acc)

        return test_y, y_predicted, acc


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def getStatistics(ms, y_label, y_predicted, acc):
    f1_score_macro = f1_score(y_label, y_predicted, average = "macro")
    f1_score_micro = f1_score(y_label, y_predicted, average = "micro")
    f1_score_weighted = f1_score(y_label, y_predicted, average = "weighted")
    f1_score_none = f1_score(y_label, y_predicted, average = None)

    true_positive = (ms[1][1] / (ms[1][1] + ms[1][0])) * 100
    true_negative = (ms[0][0] / (ms[0][0] + ms[0][1])) * 100
    false_positive = (ms[0][1] / (ms[0][0] + ms[0][1])) * 100
    false_negative = (ms[1][0] / (ms[1][0] + ms[1][1])) * 100

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    print("Precision: %.2f" % precision)
    print("Recall: %.2f" % recall)
    # print("F1 score macro: %.2f" % f1_score_macro)
    # print("F1 score micro: %.2f" % f1_score_micro)
    # print("F1 score weighted: %.2f" % f1_score_weighted)
    print("F1 score none: %.2f" % f1_score_none[1])
    print("True positive: %.2f" % true_positive + "%")
    print("True negative: %.2f" % true_negative + "%")
    print("False positive: %.2f" % false_positive + "%")
    print("False negative: %.2f" % false_negative + "%")

    with open("./_result_LSTM_inSequence/result_lstm_inSequence.csv", "w") as csvfile:
        writer = csv.writer(csvfile, delimiter = ",")

        writer.writerow(["Accuracy: %.2f" % acc])
        writer.writerow(["Precision: %.2f" % precision])
        writer.writerow(["Recall: %.2f" % recall])
        writer.writerow(["F1 score : %.2f" % f1_score_none[1]])
        writer.writerow(["True positive: %.2f" % true_positive + "%"])
        writer.writerow(["True negative: %.2f" % true_negative + "%"])
        writer.writerow(["False positive: %.2f" % false_positive + "%"])
        writer.writerow(["False negative: %.2f" % false_negative + "%"])

def draw_confusion_matrix(y_label, y_predicted, acc):
    cnf_matrix = confusion_matrix(y_label, y_predicted)

    getStatistics(cnf_matrix, y_label, y_predicted, acc)

    np.set_printoptions(precision = 2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes = "01", title = "Confusion Matrix for LSTM")
    plt.savefig("./_result_LSTM_inSequence/cm_LSTM_inSequence.png")
    # plt.show()

def draw_roc_curve(label, predicted):

    y_label = np.array(label)
    y_predicted = np.array(predicted)

    fpr, tpr, _ = roc_curve(y_label, y_predicted)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2

    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for LSTM')
    plt.legend(loc="lower right")
    plt.savefig("./_result_LSTM_inSequence/roc_lstm_inSequence.png")
    # plt.show()

def make_result(y_label, y_predicted, acc):
    test_array = np.array([0, 1])

    temp_y = []
    temp_predicted = []

    num_fraud = 0
    for i in range(len(y_label)):
        if np.array_equal(y_label[i], test_array):
            num_fraud += 1

    print(num_fraud)

    for i in range(len(y_label)):
        if np.array_equal(y_label[i], test_array):
            temp_y.append(1)
        else:
            temp_y.append(0)

    for i in range(len(y_predicted)):
        if np.array_equal(y_predicted[i], test_array):
            temp_predicted.append(1)
        else:
            temp_predicted.append(0)

    y_label = temp_y
    y_predicted = temp_predicted

    draw_confusion_matrix(y_label, y_predicted, acc)
    draw_roc_curve(y_label, y_predicted)
    saveResults(y_label, y_predicted)

def saveResults(y_label, y_predicted):
    # print(y_label)

    with open("./ResultsModelsLabels/LSTM_opt_labels.csv", "w") as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')

        for i in y_label:
            writer.writerow([str(i)])

    with open("./ResultsModelsLabels/LSTM_opt_predicted.csv", "w") as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')

        for i in y_predicted:
            writer.writerow([str(i)])


if __name__ == "__main__":
    y_label, y_predicted, acc = test()
    make_result(y_label, y_predicted, acc)
    # saveResults(y_label, y_predicted)



