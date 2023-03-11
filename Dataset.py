import numpy as np
from PIL import Image


class Dataset:
    def __init__(self):
        self.data_matrix = np.zeros(shape=(400, 10304))
        self.labels = []

    def generate_matrix(self):
        for sample_ctr in range(1, 41):
            for img_ctr in range(1, 11):
                img = Image.open('dataset/s' + str(sample_ctr) + '/' + str(img_ctr) + '.pgm')
                image_vector = np.asarray(img).flatten()
                self.data_matrix[((sample_ctr - 1) * 10) + (img_ctr - 1)] = image_vector
                self.labels.append(sample_ctr)

        return self.split_matrix(self.data_matrix, self.labels)

    def split_matrix(self, matrix, labels):
        training_set = matrix[::2, :]  # odd rows for training
        training_labels = labels[::2]
        test_set = matrix[1::2, :]  # even rows for testing
        test_labels = labels[1::2]
        return training_set, training_labels, test_set, test_labels

    def generate_split_data_70_30(self):
        # generate data
        for sample_ctr in range(1, 41):
            for img_ctr in range(1, 11):
                img = Image.open('dataset/s' + str(sample_ctr) + '/' + str(img_ctr) + '.pgm')
                image_vector = np.asarray(img).flatten()
                self.data_matrix[((sample_ctr - 1) * 10) + (img_ctr - 1)] = image_vector
                self.labels.append(sample_ctr)

        # split data
        training_set, training_labels, test_set, test_labels = [], [], [], []
        for person in range(40):
            start_idx = person * 10
            training_set.extend(self.data_matrix[start_idx:start_idx + 7])
            training_labels.extend(self.labels[start_idx:start_idx + 7])
            test_set.extend(self.data_matrix[start_idx + 7:start_idx+10])
            test_labels.extend(self.labels[start_idx + 7:start_idx + 10])

        return training_set, training_labels, test_set, test_labels


if __name__ == '__main__':
    # d = Dataset()
    # d.generate_matrix()

    X_train, y_train, X_test, y_test = Dataset().generate_split_data_70_30()
    print(np.array(X_train).shape)
    print(np.array(y_train).shape)
    print(np.array(X_test).shape)
    print(np.array(y_test).shape)
