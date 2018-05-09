
import csv
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

class readDatasetLSTMInSequence:

    fraud = []
    normal = []

    train_dataset = []
    test_dataset = []

    batch_count = 0
    _index_in_epoch = 0

    normalised = []

    train_size = 0.8

    readDataset = []

    def preprocessing(self):

        with open("one_hot_all_train.csv", "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            i = 0
            for row in reader:
                self.train_dataset.append(row)
                i += 1

        print("Trainset size: %d" % i)

        print(self.train_dataset[0])
        print(self.train_dataset[0][-1])
        # print(self.train_dataset[0])

        with open("one_hot_all_test.csv", "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            i = 0
            for row in reader:
                self.test_dataset.append(row)
                i += 1

        print("Testset size: %d" % i)

        train_dataset = self.train_dataset
        test_dataset = self.test_dataset

        num_train_set = len(train_dataset)
        num_test_set = len(test_dataset)
        print("Number of trainset: " + str(num_train_set))
        print("Number of testset: " + str(num_test_set))

        # =========================== extract train dataset ===============================
        x_train_input = []
        y_train_label = []

        scaler = MinMaxScaler()

        for i in range(len(train_dataset)):
            x_train_input.append(train_dataset[i][0:-1])
            y_train_label.append(train_dataset[i][-1])

        #################### Min Max Scaling ###################
        print(x_train_input[0])
        x_train_input = scaler.fit_transform(x_train_input)
        print(x_train_input)

        # ============================ labeling y label ===============================
        labeling = []
        for i in range(len(y_train_label)):
            if y_train_label[i] == 'normal':
                # print(y_train_label[i])
                labeling.append([1, 0])
            else:
                labeling.append([0, 1])

        temp_x_train_data = []
        for i in range(len(x_train_input)):
            temp = []
            temp.append(x_train_input[i])
            temp_x_train_data.append(temp)
            del temp

        self.x_train_data = np.array(temp_x_train_data)
        self.y_train_data = np.array(labeling)

        # print(self.x_train_data.shape)
        # print(self.y_train_data.shape)

        # =========================== extract test dataset ===============================
        x_test_input = []
        y_test_label = []
        y_test_temp = []

        for i in range(len(test_dataset)):
            x_test_input.append(test_dataset[i][0 : -1])
            y_test_label.append(test_dataset[i][-1])

        #################### Min Max Scaling ###################
        print(x_test_input[0])
        x_test_input = scaler.fit_transform(x_test_input)
        print(x_test_input[0])

        labeling = []
        for i in range(len(y_test_label)):
            if y_test_label[i] == 'normal':
                labeling.append([1, 0])
            else:
                labeling.append([0, 1])

        temp_x_test_data = []
        for i in range(len(x_test_input)):
            temp = []
            temp.append(x_test_input[i])
            temp_x_test_data.append(temp)
            del temp

        self.x_test_data = np.array(temp_x_test_data)
        self.y_test_data = np.array(labeling)

    def next_batch(self, batch_size):
        _num_examples = len(self.x_train_data)
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > _num_examples:
            start = _num_examples - batch_size
            _num_remain = self._index_in_epoch - _num_examples
            end = start + _num_remain

            self._index_in_epoch = 0

            return self.x_train_data[start:end], self.y_train_data[start:end]

        end = self._index_in_epoch

        # print("===========================================")
        # print("num_example: " + str(_num_examples) + " start: " + str(start) + " end: " + str(end))
        return self.x_train_data[start:end], self.y_train_data[start:end]

    def get_data(self):
        true = 0
        for i in range(len(self.y_train_data)):
            if np.array_equal(self.y_train_data[i], [0, 1]):
                true += 1

        print("trainset fraud: " + str(true))

        true = 0
        for i in range(len(self.y_test_data)):
            if np.array_equal(self.y_test_data[i], [0, 1]):
                true += 1

        print("testset fraud: " + str(true))

        return self.x_train_data, self.y_train_data, self.x_test_data, self.y_test_data

    def get_train_data(self):
        true = 0
        for i in range(len(self.y_train_data)):
            if np.array_equal(self.y_train_data[i], [0, 1]):
                true += 1

        print("trainset fraud: " + str(true))

        true = 0
        for i in range(len(self.y_test_data)):
            if np.array_equal(self.y_test_data[i], [0, 1]):
                true += 1

        print("testset fraud: " + str(true))

        return self.x_train_data, self.y_train_data

    def get_test_data(self):
        true = 0
        for i in range(len(self.y_train_data)):
            if np.array_equal(self.y_train_data[i], [0, 1]):
                true += 1

        print("trainset fraud: " + str(true))

        true = 0
        for i in range(len(self.y_test_data)):
            if np.array_equal(self.y_test_data[i], [0, 1]):
                true += 1

        print("testset fraud: " + str(true))

        return self.x_test_data, self.y_test_data


    def initialise(self):
        del self.x_train_data
        del self.y_train_data
        del self.x_test_data
        del self.y_test_data


if __name__ == "__main__":
    load = readDatasetLSTMInSequence()
    load.preprocessing()




