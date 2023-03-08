import pandas as pd
import numpy as np
from Dataset import Dataset
class PCA:
    def init(self):
        pass

    def projection_matrix(self,D, alpha):
        cov_matrix = np.cov(np.transpose(D), bias = True)
        eigen_values, eigen_vectors = self.__eigen_sorted(cov_matrix)
        return self.__calculate_projection_matrix(eigen_values, eigen_vectors, alpha)

    def __eigen_sorted(self,matrix):
        eigen_values, eigen_vectors = np.linalg.eigh(matrix)
        idx = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[idx]
        eigen_vectors = eigen_vectors[idx]
        return eigen_values, eigen_vectors


    def __calculate_projection_matrix(self,eigen_values, eigen_vectors, alpha):
        eigen_sum = np.sum(eigen_values)
        eigen_i = 0
        idx = -1;
        for i in eigen_values:
            print(i , ' eigh')

        for i in range(0, len(eigen_values)):
            eigen_i += eigen_values[i]
            #print(eigen_i / eigen_sum)
            if (eigen_i / eigen_sum) >= alpha:
                idx = i
                break
        return np.transpose(eigen_vectors[: idx + 1])

    def projected_data(self, P, training_set, test_set):
        return np.matmul(P, training_set), np.matmul(P, test_set)


def write_mat(mat, alpha):
    np.savetxt("PCA alpha = " + str(alpha) + ".txt", mat)


dataset = Dataset()
dataset.generate_matrix()
training_set, test_set, training_labels, test_labels = dataset.split_matrix()

pca = PCA()
alpha = [0.8,0.85,0.9,0.95]

for i in range(0, len(alpha)):
    P = pca.projection_matrix(training_set, alpha[i])
    training_set, test_set = pca.projected_data(P, training_set, test_set)


