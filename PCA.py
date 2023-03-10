import numpy as np
import pandas as pd
from Classifier import Classifier
from Dataset import Dataset


class PCA:
    def __init__(self, D):
        mu = np.mean(D, axis=0)
        Z = D - mu
        cov_matrix = np.cov(np.transpose(Z), bias=True)
        self.eigenvalues, self.eigenvectors = self.__eigen_sorted(cov_matrix)

    def __eigen_sorted(self, matrix):
        eigen_values, eigen_vectors = np.linalg.eigh(matrix)
        idx = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[idx]
        eigen_vectors = eigen_vectors[:, idx]
        return eigen_values, eigen_vectors

    def project_data(self, training_data, test_data, alpha):
        projection_matrix = self.__calculate_projection_matrix(alpha)
        return np.matmul(training_data, projection_matrix), np.matmul(test_data, projection_matrix)

    def __calculate_projection_matrix(self, alpha):
        eigen_sum = np.sum(self.eigenvalues)
        eigen_i = 0
        r = 0

        for eigenvalue in self.eigenvalues:
            r = r + 1
            eigen_i += eigenvalue
            if (eigen_i / eigen_sum) >= alpha:
                break
        return self.eigenvectors[:, :r]


def write_mat(mat, alpha):
    np.savetxt("PCA alpha = " + str(alpha) + ".txt", mat)


training_set, training_labels, test_set, test_labels = Dataset().split_matrix()
pca = PCA(training_set)
alpha = [0.8, 0.85, 0.9, 0.95]

for a in alpha:
    projected_training_set, projected_test_set = pca.project_data(training_set, test_set, a)
    neighbors = [1, 3, 5, 7]
    scores = []
    for neighbor in neighbors:
        classifier = Classifier(neighbor)
        score = classifier.classify(projected_training_set, training_labels, projected_test_set, test_labels)
        scores.append(score)
    print(f"alpha={a}:")
    df = pd.DataFrame({
        'K': neighbors,
        'scores': scores
    })
    print(df)
