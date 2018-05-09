
import tensorflow as tf
import numpy as np
import csv

# In NSL KDD dataset, there are columns 2, 3, and 4.
# These columns should be one hotting.
#
# column 2 is tcp, udp, and others
# column 3 is protocols
# column 4 is SF, things....

with open("./NSL_KDD-master/KDDTrain+.csv", "r") as csvfile:
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

    for i in range(70):
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
            for j in range(70):
                if j == 0:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)

        elif dataset[i][2] == 'private':
            for j in range(70):
                if j == 1:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)

        elif dataset[i][2] == 'domain_u':
            for j in range(70):
                if j == 2:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)

        elif dataset[i][2] == 'smtp':
            for j in range(70):
                if j == 3:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)

        elif dataset[i][2] == 'ftp_data':
            for j in range(70):
                if j == 4:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)

        elif dataset[i][2] == 'eco_i':
            for j in range(70):
                if j == 5:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'other':
            for j in range(70):
                if j == 6:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'remote_job':
            for j in range(70):
                if j == 7:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'name':
            for j in range(70):
                if j == 8:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'netbios_ns':
            for j in range(70):
                if j == 9:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'mtp':
            for j in range(70):
                if j == 10:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'telnet':
            for j in range(70):
                if j == 11:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'finger':
            for j in range(70):
                if j == 12:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'supdup':
            for j in range(70):
                if j == 13:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'uucp_path':
            for j in range(70):
                if j == 14:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'Z39_50':
            for j in range(70):
                if j == 15:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'csnet_ns':
            for j in range(70):
                if j == 16:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'uucp':
            for j in range(70):
                if j == 17:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'netbios_dgm':
            for j in range(70):
                if j == 18:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'urp_i':
            for j in range(70):
                if j == 19:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'auth':
            for j in range(70):
                if j == 20:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'domain':
            for j in range(70):
                if j == 21:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'ftp':
            for j in range(70):
                if j == 22:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'bgp':
            for j in range(70):
                if j == 23:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'ldap':
            for j in range(70):
                if j == 24:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'ecr_i':
            for j in range(70):
                if j == 25:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'gopher':
            for j in range(70):
                if j == 26:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'vmnet':
            for j in range(70):
                if j == 27:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'systat':
            for j in range(70):
                if j == 28:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'http_443':
            for j in range(70):
                if j == 29:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'efs':
            for j in range(70):
                if j == 30:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'whois':
            for j in range(70):
                if j == 31:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'imap4':
            for j in range(70):
                if j == 32:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'iso_tsap':
            for j in range(70):
                if j == 33:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'echo':
            for j in range(70):
                if j == 34:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'klogin':
            for j in range(70):
                if j == 35:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'link':
            for j in range(70):
                if j == 36:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'sunrpc':
            for j in range(70):
                if j == 37:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'login':
            for j in range(70):
                if j == 38:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'kshell':
            for j in range(70):
                if j == 39:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'sql_net':
            for j in range(70):
                if j == 40:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'time':
            for j in range(70):
                if j == 41:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'hostnames':
            for j in range(70):
                if j == 42:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'exec':
            for j in range(70):
                if j == 43:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'ntp_u':
            for j in range(70):
                if j == 44:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'discard':
            for j in range(70):
                if j == 45:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'nntp':
            for j in range(70):
                if j == 46:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'courier':
            for j in range(70):
                if j == 47:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'ctf':
            for j in range(70):
                if j == 48:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'ssh':
            for j in range(70):
                if j == 49:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'daytime':
            for j in range(70):
                if j == 50:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'shell':
            for j in range(70):
                if j == 51:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'netstat':
            for j in range(70):
                if j == 52:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'pop_3':
            for j in range(70):
                if j == 53:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'nnsp':
            for j in range(70):
                if j == 54:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'IRC':
            for j in range(70):
                if j == 55:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'pop_2':
            for j in range(70):
                if j == 56:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'printer':
            for j in range(70):
                if j == 57:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'tim_i':
            for j in range(70):
                if j == 58:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'pm_dump':
            for j in range(70):
                if j == 59:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'red_i':
            for j in range(70):
                if j == 60:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'netbios_ssn':
            for j in range(70):
                if j == 61:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'rje':
            for j in range(70):
                if j == 62:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'X11':
            for j in range(70):
                if j == 63:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'urh_i':
            for j in range(70):
                if j == 64:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'http_8001':
            for j in range(70):
                if j == 65:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'aol':
            for j in range(70):
                if j == 66:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'http_2784':
            for j in range(70):
                if j == 67:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'tftp_u':
            for j in range(70):
                if j == 68:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)
        elif dataset[i][2] == 'harvest':
            for j in range(70):
                if j == 69:
                    encoded_dataset[i].append(1)
                else:
                    encoded_dataset[i].append(0)


    print(encoded_dataset[0])
    print(encoded_dataset[1])
    print(encoded_dataset[2])
    print(encoded_dataset[3])
    print(encoded_dataset[4])

    print("===================================================================")

    print(dataset[0])
    for i in range(len(dataset)):
        if dataset[i][3] == 'SF':
            for j in [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
                encoded_dataset[i].append(j)

        elif dataset[i][3] == 'S0':
            for j in [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
                encoded_dataset[i].append(j)

        elif dataset[i][3] == 'REJ':
            for j in [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]:
                encoded_dataset[i].append(j)

        elif dataset[i][3] == 'RSTR':
            for j in [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]:
                encoded_dataset[i].append(j)

        elif dataset[i][3] == 'SH':
            for j in [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]:
                encoded_dataset[i].append(j)

        elif dataset[i][3] == 'RSTO':
            for j in [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]:
                encoded_dataset[i].append(j)

        elif dataset[i][3] == 'S1':
            for j in [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]:
                encoded_dataset[i].append(j)

        elif dataset[i][3] == 'RSTOS0':
            for j in [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]:
                encoded_dataset[i].append(j)

        elif dataset[i][3] == 'S3':
            for j in [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]:
                encoded_dataset[i].append(j)

        elif dataset[i][3] == 'S2':
            for j in [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]:
                encoded_dataset[i].append(j)

        elif dataset[i][3] == 'OTH':
            for j in [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]:
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

    with open("one_hot_all_train.csv", "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for i in range(len(encoded_dataset)):
            writer.writerow(encoded_dataset[i])


    # print(sorted_keys)









