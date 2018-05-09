
import csv
import numpy as np

with open("one_hot_train.csv", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')

    train_set = []

    for row in reader:
        temp = []

        # print(row)
        temp.append(row[0:52])

        if row[52] == 'normal':
            temp.append([1, 0])

        else:
            temp.append([0, 1])

        # i += 1
        #
        # if i == 3:
        #     break

        train_set.append(temp)
        del temp

with open("one_hot_test.csv", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')

    test_set = []

    for row in reader:
        temp = []

        temp.append(row[0:52])

        if row[52] == 'normal':
            temp.append([1, 0])
        else:
            temp.append([0, 1])

        test_set.append(temp)
        del temp


with open("train_set.csv", "w") as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    print(train_set[0])

    for i in range(len(train_set)):
        writer.writerow(train_set[i][0] + [train_set[i][1]])

with open("test_set.csv", "w") as csvfile:
    writer = csv.writer(csvfile, delimiter=',')

    for i in range(len(test_set)):
        writer.writerow(test_set[i][0] + [test_set[i][1]])


