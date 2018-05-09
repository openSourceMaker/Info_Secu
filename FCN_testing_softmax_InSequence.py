
import tensorflow as tf
import csv
import random
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, roc_curve, auc
import itertools
import readDatasetNNInSequence as load
import sys

def test(dataset):
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

    # ================== Batch Normalisation ====================
    input_layer = tf.nn.sigmoid(tf.add(tf.matmul(X, weights["W1"]), biases["b1"]))
    h1 = tf.nn.sigmoid(tf.add(tf.matmul(input_layer, weights["W2"]), biases["b2"]))
    h1 = tf.contrib.layers.batch_norm(h1, center = True, scale = True, is_training = False)

    h2 = tf.nn.relu(tf.add(tf.matmul(h1, weights["W3"]), biases["b3"]))
    h2 = tf.contrib.layers.batch_norm(h2, center = True, scale = True, is_training = False)

    h3 = tf.nn.sigmoid(tf.add(tf.matmul(h2, weights["W4"]), biases["b4"]))
    output_layer = tf.add(tf.matmul(h3, weights["W5"]), biases["b5"])

    prediction = tf.nn.softmax(output_layer)

    correction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
    acc = tf.reduce_mean(tf.cast(correction, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, "./FCN_result_inSequence/us_bank_training.ckpt")

        # extract only fraud data
        x_test_input, y_test_input = dataset.get_test_data()

        _x = x_test_input
        _y = y_test_input

        y_predicted = sess.run(tf.floor(prediction + 0.5), feed_dict = {X: _x})

        """
        true = 0
        total = len(y_predicted)
        for i in range(len(y_predicted)):
            if np.array_equal(y_predicted[i], _y[i]):
                true += 1
            # else:
                # print(y_predicted[i], _y[i])

        accuracy = (true / total) * 100
        """

        accuracy = acc.eval({X: _x, Y: _y})
        print("Accuracy: %.2f" % accuracy)
        # =========================================================

        return _y, y_predicted, accuracy

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

    with open("./_result_FCN_inSequence/result_fcn_inSequence.csv", "w") as csvfile:
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
    plot_confusion_matrix(cnf_matrix, classes = "01", title = "Confusion Matrix for FCN")
    plt.savefig("./_result_FCN_inSequence/cm_fcn_inSequence.png")
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
    plt.title('ROC for FCN')
    plt.legend(loc="lower right")
    plt.savefig("./_result_FCN_inSequence/roc_fcn_inSequence.png")
    # plt.show()

def saveResults(y_label, y_predicted):
    # print(y_label)

    with open("./ResultsModelsLabels/FCN_labels.csv", "w") as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')

        for i in y_label:
            writer.writerow([str(i)])

    with open("./ResultsModelsLabels/FCN_predicted.csv", "w") as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')

        for i in y_predicted:
            writer.writerow([str(i)])

if __name__ == "__main__":
    s_time = datetime.datetime.now()
    dataset = load.readDatasetNNInSequence()
    # dataset.split_dataset() # make k fold dataset
    dataset.preprocessing() # preprocessing for the dataset with k fold

    y_label, y_predicted, acc = test(dataset)

    test_array = np.array([0, 1])

    pre0 = []
    pre1 = []

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

    e_time = datetime.datetime.now()
    print("Training time: " + str(e_time - s_time))



