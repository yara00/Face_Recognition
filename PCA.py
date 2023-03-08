import pandas as pd
import numpy as np


def PCA(D, alpha):
    D_transpose = np.transpose(D)
    mean_vec = np.mean(D, axis=0)
    Z = D - mean_vec
    cov_matrix = np.cov(np.transpose(D))


# PCA(training_set, 0.2)


def eigen_sorted(matrix):
    eigen_values, eigen_vectors = np.linalg.eigh(matrix)
    idx = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[idx]
    eigenVectors = eigen_vectors[idx]
    return eigen_values, eigen_vectors


def reduced_basis(eigen_values, eigen_vectors, alpha):
    eigen_sum = np.sum(eigen_values)
    eigen_i = 0
    idx = -1;
    for i in range(0, len(eigen_values)):
        eigen_i += eigen_values[i]
        if (eigen_i / eigen_sum) >= alpha:
            idx = i;

    return


B = np.array([[70, -40, -15],
              [-40, 30., -20],
              [-15, -20, 930]])

eigen_values, eigen_vectors = np.linalg.eigh(B)
print(eigen_values)
print(eigen_vectors)
idx = eigen_values.argsort()[::-1]
print(idx)
eigen_values = eigen_values[idx]
eigen_vectors = eigen_vectors[idx]
print(eigen_values)
print(eigen_vectors)
