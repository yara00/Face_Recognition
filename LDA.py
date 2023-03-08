import pandas as pd
import numpy as np


class LDA:
    def __init__(self, training_set):
        self.training_set = training_set;
        self.mean_vector = np.zeros(shape=(40, 10304))

    def compute_mean_vector(self):
        sample_size = 5
        sample_sum = self.training_set.reshape(-1, sample_size, self.training_set.shape[1]).sum(axis=1)
        self.mean_vector = sample_sum / sample_size
        return self.mean_vector

    def compute_bscatter_matrix(self):
        sample_size = 5
        overall_mean = np.mean(self.training_set, axis=0)
        print(overall_mean.shape)
        Sb = 0
        for k in range(0, 40):
            Sb += sample_size * np.subtract(self.mean_vector[k], overall_mean) * np.subtract(self.mean_vector[k],
                                                                                             overall_mean).T
        return Sb
