
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

    column2 = {}
    column3 = {}
    column4 = {}

    for row in reader:
        dataset.append(row)
        if column2.get(row[1]) == None:
            column2[row[1]] = 1
        else:
            column2[row[1]] += 1

        if column3.get(row[2]) == None:
            column3[row[2]] = 1
        else:
            column3[row[2]] += 1

        if column4.get(row[3]) == None:
            column4[row[3]] = 1
        else:
            column4[row[3]] += 1

    print(column2)
    print(column3)
    print(column4)

    new_dict = {}
    sorted_dict = {}

    temp = list(column3.keys())
    temp2 = list(column3.values())

    for i in range(len(temp2)):
        new_dict[temp2[i]] = temp[i]

    print("=========================================================")
    print(new_dict)

    # keys_values = list(new_dict.keys())
    # print(keys_values)
    #
    # sorted_keys = sorted(keys_values)

    dict_sorted = sorted(new_dict, reverse=True)
    print(dict_sorted)

    final_column3 = {}

    print("")

    for i in range(len(dict_sorted)):
        final_column3[dict_sorted[i]] = new_dict[dict_sorted[i]]

    print(final_column3)

    oh_2 = []
    oh_3 = []
    oh_4 = []

    encoded_dataset = []
    print("===================================================================")
    print(dataset[0])
    print(dataset[0][2:-1])
    print("===================================================================")

    for i in range(len(dataset)): # for column 2
        encoded_dataset.append(dataset[i][0:2])

        if dataset[i][1] == 'tcp':
            encoded_dataset[i][1] = 1
            for j in [0, 0]:
                encoded_dataset[i].append(j)

            # for k in range(len(dataset[i][2:-1])):
            #     encoded_dataset[i].append(dataset[i][2 + k])

        elif dataset[i][1] == 'udp':
            encoded_dataset[i][1] = 0
            for j in [1, 0]:
                encoded_dataset[i].append(j)

            # for k in range(len(dataset[i][2:-1])):
            #     encoded_dataset[i].append(dataset[i][2 + k])


        else:
            encoded_dataset[i][1] = 0
            for j in [0, 1]:
                encoded_dataset[i].append(j)

            # for k in range(len(dataset[i][2:-1])):
            #     encoded_dataset[i].append(dataset[i][2 + k])


    print(encoded_dataset[0])
    print(encoded_dataset[1])
    print(encoded_dataset[2])
    print(encoded_dataset[3])
    print(encoded_dataset[4])
    print("===================================================================")

    print(dataset[0])
    for i in range(len(dataset)): # for column 3
        if dataset[i][2] == 'http':
            for j in [1, 0, 0, 0, 0, 0, 0]:
                encoded_dataset[i].append(j)

        elif dataset[i][2] == 'private':
            for j in [0, 1, 0, 0, 0, 0, 0]:
                encoded_dataset[i].append(j)

        elif dataset[i][2] == 'domain_u':
            for j in [0, 0, 1, 0, 0, 0, 0]:
                encoded_dataset[i].append(j)

        elif dataset[i][2] == 'smtp':
            for j in [0, 0, 0, 1, 0, 0, 0]:
                encoded_dataset[i].append(j)

        elif dataset[i][2] == 'ftp_data':
            for j in [0, 0, 0, 0, 1, 0, 0]:
                encoded_dataset[i].append(j)

        elif dataset[i][2] == 'eco_i':
            for j in [0, 0, 0, 0, 0, 1, 0]:
                encoded_dataset[i].append(j)

        else:
            for j in [0, 0, 0, 0, 0, 0, 1]:
                encoded_dataset[i].append(j)


    print(encoded_dataset[0])
    print(encoded_dataset[1])
    print(encoded_dataset[2])
    print(encoded_dataset[3])
    print(encoded_dataset[4])

    print("===================================================================")

    print(dataset[0])
    for i in range(len(dataset)): # for column 4
        if dataset[i][3] == 'SF':
            for j in [1, 0, 0, 0]:
                encoded_dataset[i].append(j)

        elif dataset[i][3] == 'S0':
            for j in [0, 1, 0, 0]:
                encoded_dataset[i].append(j)

        elif dataset[i][3] == 'REJ':
            for j in [0, 0, 1, 0]:
                encoded_dataset[i].append(j)

        else:
            for j in [0, 0, 0, 1]:
                encoded_dataset[i].append(j)

    print(encoded_dataset[0])
    print(encoded_dataset[1])
    print(encoded_dataset[2])
    print(encoded_dataset[3])
    print(encoded_dataset[4])

    print("===================================================================")

    for i in range(len(dataset)): # for all
        for j in range(len(dataset[i][4:-1])):
            encoded_dataset[i].append(dataset[i][4 + j])

    print(encoded_dataset[0])
    print(encoded_dataset[1])
    print(encoded_dataset[2])
    print(encoded_dataset[3])
    print(encoded_dataset[4])

    print(len(dataset))
    print(len(encoded_dataset[0]))
    print(len(encoded_dataset))

    with open("one_hot_test.csv", "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for i in range(len(encoded_dataset)):
            writer.writerow(encoded_dataset[i])

    # print(sorted_keys)









