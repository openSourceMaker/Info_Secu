
from sklearn import svm
import numpy as np
import datetime
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
import csv
import sys

clf = svm.SVC()

def loadData():

    train_x = []
    train_y = []
    test_x = []
    test_y = []

    with open("one_hot_all_train.csv", "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        for row in reader:
            # print(row)
            train_x.append(row[0:-1])

            if row[-1] == 'normal':
                train_y.append(0)

            else:
                train_y.append(1)

    with open("one_hot_all_test.csv", "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        for row in reader:
            # print(row)
            test_x.append(row[0:-1])

            if row[-1] == 'normal':
                test_y.append(0)

            else:
                test_y.append(1)

    return train_x, train_y, test_x, test_y


def train(train_x, train_y):
    x_train_input = train_x
    y_train_input = train_y

    clf.fit(x_train_input, y_train_input)

    true = 0
    predicted = clf.predict(x_train_input)

    for i in range(len(y_train_input)):
        if predicted[i] == y_train_input[i]:
            true += 1

    total = len(y_train_input)
    acc = (true / total) * 100
    print("Trained accuracy: " + "%.2f" % acc + "%")


def test(test_x, test_y):
    x_test_input = test_x
    y_test_input = test_y

    true = 0

    # print(y_test)
    predicted = clf.predict(x_test_input)
    return_predicted = predicted
    return_y_label = y_test_input
    for i in range(len(y_test_input)):
        if predicted[i] == y_test_input[i]:
            true += 1

    total = len(y_test_input)
    acc = (true / total) * 100
    print("Predicted accuracy: " + "%.2f" % acc + "%")

    return return_y_label, return_predicted, acc


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

    with open("./_result_SVM_inSequence/result_svm_inSequence.csv", "w") as csvfile:
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
    plot_confusion_matrix(cnf_matrix, classes = "01", title = "Confusion Matrix for SVM")
    plt.savefig("./_result_SVM_inSequence/cm_svm_inSequence.png")
    plt.show()

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
    plt.title('ROC for SVM')
    plt.legend(loc="lower right")
    plt.savefig("./_result_SVM_inSequence/roc_svm_inSequence.png")
    plt.show()

def saveResults(y_label, y_predicted):
    # print(y_label)

    with open("./ResultsModelsLabels/SVM_labels.csv", "w") as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')

        for i in y_label:
            writer.writerow(str(i))

    with open("./ResultsModelsLabels/SVM_predicted.csv", "w") as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')

        for i in y_predicted:
            writer.writerow(str(i))


if __name__ == "__main__":

    train_x, train_y, test_x, test_y = loadData()

    s_time = datetime.datetime.now()
    train(train_x, train_y)
    e_time = datetime.datetime.now()

    y_label, y_predicted, acc = test(test_x, test_y)

    draw_confusion_matrix(y_label, y_predicted, acc)
    draw_roc_curve(y_label, y_predicted)
    saveResults(y_label, y_predicted)

    print("Training time: " + str(e_time - s_time))



