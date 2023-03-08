import pandas as pd
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
        return self.data_matrix, self.labels

    def split_matrix(self):
        self.generate_matrix()
        training_set = self.data_matrix[::2, :]  # odd rows for training
        training_labels = self.labels[::2]
        test_set = self.data_matrix[1::2, :]  # even rows for testing
        test_labels = self.labels[1::2]
        return training_set, training_labels, test_set, test_labels



