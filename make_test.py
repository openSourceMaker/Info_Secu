
import tensorflow as tf
import numpy as np
import csv

# In NSL KDD dataset, there are columns 2, 3, and 4.
# These columns should be one hotting.
#
# column 2 is tcp, udp, and others
# column 3 is protocols
# column 4 is SF, things....

with open("./NSL_KDD-master/KDDTest+.csv", "r") as csvfile:
    reader = csv.reader(csvfile)

    dataset = []

    i = 0
    for row in reader:
        temp = []
        temp.append(row[0])
        for element in row[4:42]:
            temp.append(element)

        dataset.append(temp)

        # print(row)
        #
        # i += 1
        #
        # if i == 3:
        #     break

        del temp


    # index 41 is the label
    # index 1, 2, 3 should be excluded
    print(len(dataset[0]))
    # print(np.array(dataset))

    dataset = np.array(dataset)

    with open("numeric_test.csv", "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for i in range(len(dataset)):
            writer.writerow(dataset[i])

