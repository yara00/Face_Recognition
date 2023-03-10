import numpy as np


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
